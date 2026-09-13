package dev.hamster.vda.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

/**
 * Material You dynamic colour is deliberately OFF. The app's palette is a fixed brand scheme
 * because (a) "one common theme across the app" is not achievable from a per-device wallpaper,
 * (b) the app's own content is either achromatic (the grayscale depth map) or on a fixed
 * scientific colormap, neither of which should be re-tinted per device, so the chrome palette
 * is the only colour there is and must be chosen, not inherited, and (c) every contrast
 * ratio this design depends on was measured against these exact values.
 * Flip this to true only to A/B the wallpaper-derived palette on a device; nothing in the app's
 * visual language depends on it and there is no user-facing toggle.
 */
private const val USE_DYNAMIC_COLOR = false

private val VdaLightColors = lightColorScheme(
    primary = VdaPrimary40,
    onPrimary = VdaPrimary100,
    primaryContainer = VdaPrimary90,
    onPrimaryContainer = VdaPrimary10,
    inversePrimary = VdaPrimary80,
    secondary = VdaSecondary40,
    onSecondary = VdaSecondary100,
    secondaryContainer = VdaSecondary90,
    onSecondaryContainer = VdaSecondary10,
    tertiary = VdaTertiary40,
    onTertiary = VdaTertiary100,
    tertiaryContainer = VdaTertiary90,
    onTertiaryContainer = VdaTertiary10,
    background = VdaNeutral98,
    onBackground = VdaNeutral10,
    surface = VdaNeutral98,
    onSurface = VdaNeutral10,
    surfaceVariant = VdaNeutralVariant90,
    onSurfaceVariant = VdaNeutralVariant30,
    inverseSurface = VdaNeutral20,
    inverseOnSurface = VdaNeutral95,
    outline = VdaNeutralVariant50,
    outlineVariant = VdaNeutralVariant80,
    scrim = VdaNeutral0,
    surfaceBright = VdaNeutral98,
    surfaceDim = VdaNeutral87,
    surfaceContainerLowest = VdaNeutral100,
    surfaceContainerLow = VdaNeutral96,
    surfaceContainer = VdaNeutral94,
    surfaceContainerHigh = VdaNeutral92,
    surfaceContainerHighest = VdaNeutral90,
)

private val VdaDarkColors = darkColorScheme(
    primary = VdaPrimary80,
    onPrimary = VdaPrimary20,
    primaryContainer = VdaPrimary30,
    onPrimaryContainer = VdaPrimary90,
    inversePrimary = VdaPrimary40,
    secondary = VdaSecondary80,
    onSecondary = VdaSecondary20,
    secondaryContainer = VdaSecondary30,
    onSecondaryContainer = VdaSecondary90,
    tertiary = VdaTertiary80,
    onTertiary = VdaTertiary20,
    tertiaryContainer = VdaTertiary30,
    onTertiaryContainer = VdaTertiary90,
    background = VdaNeutral6,
    onBackground = VdaNeutral90,
    surface = VdaNeutral6,
    onSurface = VdaNeutral90,
    surfaceVariant = VdaNeutralVariant30,
    onSurfaceVariant = VdaNeutralVariant80,
    inverseSurface = VdaNeutral90,
    inverseOnSurface = VdaNeutral20,
    outline = VdaNeutralVariant60,
    outlineVariant = VdaNeutralVariant30,
    scrim = VdaNeutral0,
    surfaceBright = VdaNeutral24,
    surfaceDim = VdaNeutral6,
    surfaceContainerLowest = VdaNeutral4,
    surfaceContainerLow = VdaNeutral10,
    surfaceContainer = VdaNeutral12,
    surfaceContainerHigh = VdaNeutral17,
    surfaceContainerHighest = VdaNeutral22,
)

@Composable
fun VdaTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val colorScheme = when {
        USE_DYNAMIC_COLOR && dark -> dynamicDarkColorScheme(context)
        USE_DYNAMIC_COLOR -> dynamicLightColorScheme(context)
        dark -> VdaDarkColors
        else -> VdaLightColors
    }
    MaterialTheme(
        colorScheme = colorScheme,
        typography = VdaTypography,
        shapes = VdaShapes,
        content = content
    )
}
