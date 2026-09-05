"""Issue short-lived, fixed-file upload capabilities; keep drive credentials server-side."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import posixpath
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BrokerError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


class OwnedRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    secret: str = Field(min_length=32, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')


class Completion(OwnedRequest):
    revision: int = Field(ge=1, le=2147483647)


class Submission(Completion):
    sample_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    species: list[str] = Field(min_length=1, max_length=8)
    image_bytes: int = Field(ge=1, le=16 * 1024 * 1024)
    annotation_bytes: int = Field(ge=1, le=1024 * 1024)

    @field_validator('species')
    @classmethod
    def safe_species(cls, values):
        if len({unicodedata.normalize('NFC', x).casefold() for x in values}) != len(values):
            raise ValueError('duplicate species')
        for name in values:
            if (not 1 <= len(name) <= 80 or name != name.strip() or name.endswith('.')
                    or re.search(r'[\\/:*?"<>|%\x00-\x1f\x7f]', name) or name in ('.', '..')):
                raise ValueError('invalid species name')
        return values


@dataclass
class BrokerConfig:
    state_dir: Path
    root_path: str = '/Neri_Data/Training'
    openlist_url: str = 'http://127.0.0.1:5244'
    openlist_token: str = ''
    public_url: str = 'https://myneri.top/api/training'
    requests_per_minute: int = 60
    daily_ip_bytes: int = 512 * 1024 * 1024
    daily_total_bytes: int = 2 * 1024 * 1024 * 1024

    @classmethod
    def from_env(cls):
        return cls(Path(os.environ.get('NERI_TRAINING_STATE', '/var/lib/neri-training')),
                   openlist_token=os.environ['NERI_OPENLIST_TOKEN'],
                   daily_ip_bytes=int(os.environ.get('NERI_DAILY_IP_BYTES', 512 * 1024 * 1024)),
                   daily_total_bytes=int(os.environ.get('NERI_DAILY_TOTAL_BYTES', 2 * 1024 ** 3)))


def valid_upload_url(url):
    try:
        p = urllib.parse.urlsplit(url)
        host = (p.hostname or '').lower()
        return (p.scheme == 'https' and not p.username and not p.password and p.port in (None, 443)
                and (host == 'my.microsoftpersonalcontent.com' or host.endswith('.1drv.com')
                     or host.endswith('.sharepoint.com')))
    except (TypeError, ValueError):
        return False


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OpenListDrive:
    def __init__(self, config):
        self.config = config
        self.opener = urllib.request.build_opener(NoRedirect)

    def _call(self, endpoint, body, headers=None, missing_ok=False):
        request = urllib.request.Request(self.config.openlist_url + '/api/fs/' + endpoint,
            data=json.dumps(body).encode(), headers={'Authorization': self.config.openlist_token,
            'Content-Type': 'application/json', **(headers or {})}, method='POST')
        try:
            with self.opener.open(request, timeout=30) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError()
            value = json.loads(raw)
        except (OSError, ValueError):
            raise BrokerError(502, 'drive_unavailable') from None
        if value.get('code') != 200:
            message = str(value.get('message', '')).lower()
            if missing_ok and ('not found' in message or '不存在' in message):
                return None
            raise BrokerError(502, 'drive_operation_failed')
        return value.get('data')

    def mkdir(self, path):
        if self.stat(path) is None:
            self._call('mkdir', {'path': path})

    def stat(self, path):
        # fs/get ignores refresh; refresh the parent listing to invalidate OpenList's
        # directory cache after direct Graph writes before looking up the exact item.
        self._call('list', {'path': posixpath.dirname(path), 'password': '', 'page': 1,
                           'per_page': 1, 'refresh': True}, missing_ok=True)
        value = self._call('get', {'path': path, 'password': ''}, missing_ok=True)
        return None if value is None else {'size': value['size'], 'is_dir': value['is_dir']}

    def create_session(self, path, size):
        result = self._call('get_direct_upload_info', {'path': posixpath.dirname(path),
            'file_name': posixpath.basename(path), 'file_size': size, 'tool': 'HttpDirect'},
            {'File-Path': urllib.parse.quote(path, safe=''), 'Overwrite': 'true'})
        if not valid_upload_url(result.get('upload_url')) or result.get('method', 'PUT') != 'PUT':
            raise BrokerError(502, 'invalid_drive_capability')
        return {'upload_url': result['upload_url'], 'chunk_size': 5242880}

    def delete(self, path):
        if self.stat(path) is not None:
            self._call('remove', {'dir': posixpath.dirname(path), 'names': [posixpath.basename(path)]})

    def cancel(self, url):
        if not valid_upload_url(url):
            raise BrokerError(502, 'invalid_drive_capability')
        try:
            with self.opener.open(urllib.request.Request(url, method='DELETE'), timeout=20):
                pass
        except urllib.error.HTTPError as error:
            if error.code not in (404, 410):
                raise BrokerError(502, 'cancel_failed') from None
        except OSError:
            raise BrokerError(502, 'cancel_failed') from None



class Broker:
    def __init__(self, config, drive, clock=time.time):
        self.config, self.drive, self.clock = config, drive, clock
        self.lock = threading.RLock()
        config.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = config.state_dir / 'broker.sqlite3'
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY,owner TEXT NOT NULL,
                    latest INTEGER NOT NULL,completed INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL,updated REAL NOT NULL,deleted INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS revisions(id TEXT NOT NULL,rev INTEGER NOT NULL,
                    signature TEXT NOT NULL,targets TEXT NOT NULL,complete INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL,PRIMARY KEY(id,rev));
                CREATE TABLE IF NOT EXISTS budgets(key TEXT PRIMARY KEY,amount INTEGER NOT NULL);
            ''')
            db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)', ('ip_salt', secrets.token_hex(32)))
            self.salt = db.execute('SELECT value FROM settings WHERE key=?', ('ip_salt',)).fetchone()[0]
        if os.name != 'nt':
            config.state_dir.chmod(0o700)
            self.db_path.chmod(0o600)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.db_path, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                yield db
        finally:
            db.close()

    @contextmanager
    def operation(self):
        # Keep the interprocess lock across explicit commits and remote calls.
        with self.lock, open(self.config.state_dir / 'operation.lock', 'a+b') as handle:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                with self.db() as db:
                    yield db
            finally:
                handle.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def owner(secret):
        return hashlib.sha256(secret.encode()).hexdigest()

    def authorize(self, db, sample_id, secret):
        sample = db.execute('SELECT * FROM samples WHERE id=?', (sample_id,)).fetchone()
        if sample is None or not hmac.compare_digest(sample['owner'], self.owner(secret)):
            raise BrokerError(403, 'sample_access_denied')
        return sample

    def budget(self, ip, size=0):
        # Persist counters separately so failed authorization/drive requests still consume rate budget.
        ip_hash = hmac.new(self.salt.encode(), ip.encode(), hashlib.sha256).hexdigest()
        minute, day = int(self.clock() // 60), int(self.clock() // 86400)
        limits = [(f'm:{minute}:{ip_hash}', 1, self.config.requests_per_minute)]
        if size:
            limits += [(f'd:{day}:{ip_hash}', size, self.config.daily_ip_bytes),
                       (f'd:{day}:total', size, self.config.daily_total_bytes)]
        with self.db() as db:
            for key, amount, limit in limits:
                row = db.execute('SELECT amount FROM budgets WHERE key=?', (key,)).fetchone()
                if (row['amount'] if row else 0) + amount > limit:
                    raise BrokerError(429, 'training_upload_limit')
            for key, amount, _ in limits:
                db.execute('INSERT INTO budgets VALUES (?,?) ON CONFLICT(key) DO UPDATE SET amount=amount+excluded.amount', (key, amount))
            db.execute("DELETE FROM budgets WHERE (key LIKE 'm:%' AND CAST(substr(key,3,instr(substr(key,3),':')-1) AS INTEGER)<?) OR (key LIKE 'd:%' AND CAST(substr(key,3,instr(substr(key,3),':')-1) AS INTEGER)<?)", (minute - 2, day - 1))

    def _cancel(self, targets):
        for target in targets:
            if target.get('remote_url'):
                self.drive.cancel(target['remote_url'])

    def _remove(self, targets):
        self._cancel(targets)
        for target in targets:
            self.drive.delete(target['path'])

    def _public_targets(self, targets):
        return [{'kind': t['kind'], 'species': t['species'], 'chunk_size': 5242880,
                 'upload_url': t['remote_url'],
                 'cancel_url': self.config.public_url.rstrip('/') + '/v1/uploads/' + t['token']} for t in targets]

    def create(self, data, ip):
        self.budget(ip, (data.image_bytes + data.annotation_bytes) * len(data.species))
        signature = json.dumps([sorted(data.species), data.image_bytes, data.annotation_bytes])
        with self.operation() as db:
            sample = db.execute('SELECT * FROM samples WHERE id=?', (data.sample_id,)).fetchone()
            if sample:
                self.authorize(db, data.sample_id, data.secret)
                if data.revision < sample['latest'] or (sample['deleted'] and data.revision <= sample['latest']):
                    raise BrokerError(409, 'stale_revision')
            else:
                db.execute('INSERT INTO samples(id,owner,latest,created,updated) VALUES (?,?,?,?,?)',
                           (data.sample_id, self.owner(data.secret), data.revision, self.clock(), self.clock()))
            old = db.execute('SELECT * FROM revisions WHERE id=? AND rev=?', (data.sample_id, data.revision)).fetchone()
            response = {'sample_id': data.sample_id, 'revision': data.revision, 'targets': []}
            if old and old['signature'] != signature:
                raise BrokerError(409, 'revision_payload_changed')
            if old and old['complete']:
                return {**response, 'already_complete': True}
            # Remove abandoned attempts, including fully uploaded files. Completed revisions stay until commit.
            for row in db.execute('SELECT targets FROM revisions WHERE id=? AND complete=0', (data.sample_id,)):
                self._remove(json.loads(row['targets']))
            db.execute('DELETE FROM revisions WHERE id=? AND complete=0', (data.sample_id,))
            targets = []
            attempt = secrets.token_hex(8)
            for species in data.species:
                folder = self.config.root_path.rstrip('/') + '/' + species
                for kind, suffix, size in [('image', '.jpg', data.image_bytes), ('annotation', '.json', data.annotation_bytes)]:
                    path = f'{folder}/{data.sample_id}-r{data.revision}-{attempt}{suffix}'
                    targets.append({'kind': kind, 'species': species, 'path': path, 'size': size,
                                    'token': secrets.token_urlsafe(32)})
            db.execute('INSERT OR REPLACE INTO revisions VALUES (?,?,?,?,0,?)',
                       (data.sample_id, data.revision, signature, json.dumps(targets), self.clock()))
            db.execute('UPDATE samples SET latest=?,updated=?,deleted=0 WHERE id=?', (data.revision, self.clock(), data.sample_id))
            db.commit()  # Track every unique destination before issuing a capability.
            for target in targets:
                self.drive.mkdir(posixpath.dirname(target['path']))
                capability = self.drive.create_session(target['path'], target['size'])
                if not valid_upload_url(capability.get('upload_url')):
                    raise BrokerError(502, 'invalid_drive_capability')
                target['remote_url'] = capability['upload_url']
                db.execute('UPDATE revisions SET targets=? WHERE id=? AND rev=?',
                           (json.dumps(targets), data.sample_id, data.revision))
                db.commit()  # A crash can leave only an empty session, never untracked uploaded bytes.
            response['targets'] = self._public_targets(targets)
            return response

    def cancel_upload(self, token, ip):
        self.budget(ip)
        if not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
            raise BrokerError(403, 'upload_access_denied')
        with self.operation() as db:
            row = db.execute('SELECT * FROM revisions WHERE complete=0 AND targets LIKE ?', ('%' + token + '%',)).fetchone()
            if row is None or not any(hmac.compare_digest(t['token'], token) for t in json.loads(row['targets'])):
                raise BrokerError(403, 'upload_access_denied')
            self._remove(json.loads(row['targets']))
            db.execute('DELETE FROM revisions WHERE id=? AND rev=?', (row['id'], row['rev']))
        return {'status': 'cancelled'}

    def complete(self, sample_id, revision, secret, ip):
        self.budget(ip)
        with self.operation() as db:
            sample = self.authorize(db, sample_id, secret)
            row = db.execute('SELECT * FROM revisions WHERE id=? AND rev=?', (sample_id, revision)).fetchone()
            if sample['deleted'] or sample['latest'] != revision or row is None:
                raise BrokerError(409, 'stale_revision')
            for target in json.loads(row['targets']):
                value = self.drive.stat(target['path'])
                if value is None or value['is_dir']:
                    raise BrokerError(409, 'upload_incomplete')
                if value['size'] != target['size']:
                    self._remove(json.loads(row['targets']))
                    db.execute('DELETE FROM revisions WHERE id=? AND rev=?', (sample_id, revision))
                    db.commit()
                    raise BrokerError(413, 'uploaded_size_mismatch')
            for old in db.execute('SELECT targets FROM revisions WHERE id=? AND rev<>?', (sample_id, revision)):
                self._remove(json.loads(old['targets']))
            db.execute('DELETE FROM revisions WHERE id=? AND rev<>?', (sample_id, revision))
            db.execute('UPDATE revisions SET complete=1 WHERE id=? AND rev=?', (sample_id, revision))
            db.execute('UPDATE samples SET completed=?,updated=? WHERE id=?', (revision, self.clock(), sample_id))
            return {'status': 'complete'}

    def delete(self, sample_id, secret, ip):
        self.budget(ip)
        with self.operation() as db:
            self.authorize(db, sample_id, secret)
            self._delete(db, sample_id)
        return {'status': 'deleted'}

    def _delete(self, db, sample_id):
        for row in db.execute('SELECT targets FROM revisions WHERE id=?', (sample_id,)):
            self._remove(json.loads(row['targets']))
        db.execute('DELETE FROM revisions WHERE id=?', (sample_id,))
        db.execute('UPDATE samples SET deleted=1,completed=0,updated=? WHERE id=?', (self.clock(), sample_id))

    def cleanup(self):
        with self.operation() as db:
            # An abandoned revision can already contain a finished JPEG: delete files as well as sessions.
            expired = db.execute('SELECT * FROM revisions WHERE (complete=0 AND created<?) OR created<?',
                                (self.clock() - 1200, self.clock() - 365 * 86400)).fetchall()
            for row in expired:
                self._remove(json.loads(row['targets']))
                db.execute('DELETE FROM revisions WHERE id=? AND rev=?', (row['id'], row['rev']))
                db.execute('UPDATE samples SET completed=0 WHERE id=? AND completed=?', (row['id'], row['rev']))


def create_app(config=None, drive=None):
    config = config or BrokerConfig.from_env()
    broker = Broker(config, drive or OpenListDrive(config))
    stop = threading.Event()

    def clean_loop():
        while not stop.is_set():
            try:
                broker.cleanup()
            except Exception:
                pass  # Retry every five minutes; never log capability URLs or request secrets.
            stop.wait(300)

    @asynccontextmanager
    async def lifespan(app):
        worker = threading.Thread(target=clean_loop, name='training-retention', daemon=True)
        worker.start()
        yield
        stop.set()
        worker.join(timeout=2)

    app = FastAPI(title='Neri training upload', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.exception_handler(BrokerError)
    async def broker_error(request, error):
        return JSONResponse({'error': error.message}, status_code=error.status)

    @app.middleware('http')
    async def native_only(request, call_next):
        if request.headers.get('origin'):
            return JSONResponse({'error': 'native_client_required'}, status_code=403)
        return await call_next(request)

    def client_ip(request):
        peer = request.client.host if request.client else 'unknown'
        return request.headers.get('x-real-ip', peer) if peer in ('127.0.0.1', '::1') else peer

    @app.get('/health')
    def health():
        return {'status': 'ok', 'schema_version': 1}

    @app.post('/v1/submissions')
    def create(data: Submission, request: Request):
        return broker.create(data, client_ip(request))

    @app.delete('/v1/uploads/{token}')
    def cancel_upload(token: str, request: Request):
        return broker.cancel_upload(token, client_ip(request))

    @app.post('/v1/submissions/{sample_id}/complete')
    def complete(sample_id: str, data: Completion, request: Request):
        return broker.complete(sample_id, data.revision, data.secret, client_ip(request))

    @app.delete('/v1/submissions/{sample_id}')
    def delete(sample_id: str, data: OwnedRequest, request: Request):
        return broker.delete(sample_id, data.secret, client_ip(request))

    return app
