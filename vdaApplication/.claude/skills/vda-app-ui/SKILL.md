---
name: vda-app-ui
description: Compose UI conventions for the VDA app -- the fixed mustard brand theme and how its palette is generated, screen/navigation structure, and the reusable card, badge, sheet and toast patterns to copy when adding UI. Use when adding a screen or control, restyling anything, changing the palette, or wondering why dynamic colour and Material defaults are deliberately overridden.
---

# VDA app: UI conventions

All UI is Jetpack Compose under `ui/`.

## Theme -- fixed, not Material You

`ui/theme/Theme.kt` sets `USE_DYNAMIC_COLOR = false` **on purpose**:

- one look across devices is not achievable from a per-device wallpaper,
- the app's own content is a grayscale map or a fixed scientific colormap,
  which must never be re-tinted, so the chrome palette is the only colour
  there is and must be chosen,
- every contrast ratio in the design was measured against these exact values.

Do not reintroduce `dynamicLightColorScheme`/`dynamicDarkColorScheme`, and
**always use `MaterialTheme.colorScheme` roles, never literal colours**.

### The palette is generated, not hand-edited

`ui/theme/Color.kt` holds tonal ramps at **CIE LCh hue 84 (mustard)** for
primary/secondary/neutrals and **hue 250 (slate blue)** for tertiary. Tone ==
CIE L*; chroma is reduced per tone only as far as sRGB requires. To change the
hue, **regenerate the whole ramp** and re-check every pair the two schemes use
(light: `primary/onPrimary`, `*Container/on*Container`, `surface/onSurface`,
`surfaceVariant/onSurfaceVariant`; dark: the same at the dark tones) for
>= 4.5:1. The generator is a few lines of Python (Lab -> sRGB with chroma
clipping and a WCAG check); the ramp comment in `Color.kt` records the
parameters. Then keep in sync:

- `res/values/colors.xml` = Neutral98, `res/values-night/colors.xml` = Neutral6
  (the window background behind the first frame).
- `res/drawable/ic_launcher_background.xml` = Primary30 and
  `ic_launcher_foreground.xml` = the logo mark in Primary90 -- the launcher
  icon *is* the brand mark, same two tones as the dark scheme's primary
  container.

`ic_vda_logo.xml` is single-colour with alpha-only variation on purpose: it
doubles as the launcher's `monochrome` layer and tints correctly everywhere.

`VdaShapes` (10/16/20/28 dp) restyles cards/sheets/surfaces. Buttons stay pills
regardless: `ButtonDefaults.shape` resolves to `CircleShape` independently of
`MaterialTheme.shapes`.

## Navigation

There is **no `androidx.navigation`**. `MainActivity` holds
`private enum class VdaDestination { HOME, DEPTH, MODELS }` in `rememberSaveable`,
with `BackHandler(enabled = destination != HOME)`. Add a destination by
extending the enum and the `when`.

`MainActivity` collects `uiState` eagerly so `DepthViewModel` is constructed at
launch and its model load starts in the background, and calls
`ModelRepository.get(this).ensureBootstrapModels()` first thing. Do not
"optimise" either away.

## Patterns to copy

**Tappable card** (`HomeScreen.HomeActionTile`, `ModelsScreen.ModelCard`):

```kotlin
Surface(
    onClick = ..., enabled = ...,
    shape = MaterialTheme.shapes.extraLarge,
    color = containerColor, contentColor = contentColor,
    modifier = Modifier.fillMaxWidth()
) {
    Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(16.dp)) { ... }
}
```

Subtitles use `bodySmall` at `LocalContentColor.current.copy(alpha = 0.8f)`.

**Badge chip**: `Surface(shape = CircleShape, color = surfaceContainerHighest,
contentColor = onSurfaceVariant)` with `labelSmall` at 10 dp / 4 dp padding.

**State by colour**: an installed/selected item switches **both** container and
content colour (`primaryContainer`/`onPrimaryContainer`); failures use
`errorContainer`/`onErrorContainer`. Keep trailing elements the same width
across sibling cards so rows stay the same height.

**Messages**: transient feedback from the ViewModel goes through
`DepthUiState.transientMessage` -> Snackbar. Direct confirmations from a
callback use `Toast`. Match the surrounding code.

## Hard rules

- **No hardcoded UI strings.** Everything goes in `res/values/strings.xml`.
- **Use `ButtonDefaults` constants**, never dp literals for button metrics.
- **`ConfigSheet`'s content column is `verticalScroll`ed -- keep it.** VDA's
  sheet has five dropdowns and a two-line model-file block; adding rows once
  pushed Apply/Cancel below the fold on a 20:9 phone with no way to reach
  them, which made every setting unappliable. Any new row goes inside that
  scrolling column.
- **`ConfigSheet` narrows source -> resolution -> backbone -> input size ->
  window**, each `LaunchedEffect` snapping a stale selection back in range. A
  new dimension must join that chain, in `ModelCatalog` too, and must handle
  the list going empty (nothing installed).
- **`Material3 SegmentedButton` hardcodes `Modifier.weight(1f)`** per segment.
  `DepthScreen.OutputKindRow` is a hand-rolled replacement using the public
  `SegmentedButtonDefaults.itemShape`/`.BorderWidth`; extend it rather than
  fighting the stock one.
- Screenshots for docs live in `docs/images/` at 280 px wide with the status
  bar cropped (see `vda-app-verify` for the recipe) and must not include
  personal gallery content -- use the converter's public example clip.
