const closeBehaviorSettingKey = 'close_behavior';
const closeBehaviorAsk = 'ask';
const closeBehaviorHideToTray = 'hide_to_tray';
const closeBehaviorExit = 'exit';

String normalizeCloseBehavior(Object? value) {
  return switch (value) {
    closeBehaviorHideToTray => closeBehaviorHideToTray,
    closeBehaviorExit => closeBehaviorExit,
    _ => closeBehaviorAsk,
  };
}
