const previewPagePadding = 16.0;
const previewWorkspaceDividerWidth = 24.0;

double previewLeadingPaneWidth(double availableWidth) {
  return (availableWidth * 0.20).clamp(200.0, 300.0).toDouble();
}

double previewDistanceToLeadingDivider(double pageWidth) {
  final workspaceWidth = pageWidth > previewPagePadding * 2
      ? pageWidth - previewPagePadding * 2
      : 0.0;
  return previewPagePadding +
      previewLeadingPaneWidth(workspaceWidth) +
      previewWorkspaceDividerWidth / 2;
}
