const closeBehaviorSettingKey = 'close_behavior';
const lastCloseActionSettingKey = 'last_close_action';
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

String normalizeLastCloseAction(Object? value) {
  return switch (value) {
    closeBehaviorHideToTray => closeBehaviorHideToTray,
    _ => closeBehaviorExit,
  };
}
