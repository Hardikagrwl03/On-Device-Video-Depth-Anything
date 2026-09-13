package dev.hamster.vda.ui.theme

import androidx.compose.ui.graphics.Color

// Brand tonal ramps. CIE LCh hue 84 (mustard - the hue of #E1AD01) for primary/secondary/
// neutrals, hue 250 (slate blue) for tertiary. Tone == CIE L*; chroma is reduced per tone
// only as far as needed to stay inside sRGB, which is how Material builds a tonal palette.
// Generated once and verified for WCAG contrast on every pair the two colour schemes below
// actually use (all >= 4.5:1; see the light/dark scheme definitions in Theme.kt). Do not
// hand-edit individual values - regenerate the whole ramp if the hue ever changes.
internal val VdaPrimary10 = Color(0xFF241A00)
internal val VdaPrimary20 = Color(0xFF3F2E00)
internal val VdaPrimary30 = Color(0xFF5B4301)
internal val VdaPrimary40 = Color(0xFF785A00)
internal val VdaPrimary80 = Color(0xFFF1BF4E)
internal val VdaPrimary90 = Color(0xFFFFDEA5)
internal val VdaPrimary100 = Color(0xFFFFFFFF)

internal val VdaSecondary10 = Color(0xFF241A00)
internal val VdaSecondary20 = Color(0xFF3C2E0F)
internal val VdaSecondary30 = Color(0xFF554424)
internal val VdaSecondary40 = Color(0xFF6E5C3B)
internal val VdaSecondary80 = Color(0xFFDAC49E)
internal val VdaSecondary90 = Color(0xFFF7DFB9)
internal val VdaSecondary100 = Color(0xFFFFFFFF)

internal val VdaTertiary10 = Color(0xFF011E2C)
internal val VdaTertiary20 = Color(0xFF01344A)
internal val VdaTertiary30 = Color(0xFF014C6A)
internal val VdaTertiary40 = Color(0xFF236486)
internal val VdaTertiary80 = Color(0xFF95CDF3)
internal val VdaTertiary90 = Color(0xFFC7E7FF)
internal val VdaTertiary100 = Color(0xFFFFFFFF)

internal val VdaNeutral0 = Color(0xFF000000)
internal val VdaNeutral4 = Color(0xFF120E05)
internal val VdaNeutral6 = Color(0xFF17130C)
internal val VdaNeutral10 = Color(0xFF1E1B16)
internal val VdaNeutral12 = Color(0xFF221F1A)
internal val VdaNeutral17 = Color(0xFF2D2924)
internal val VdaNeutral20 = Color(0xFF33302B)
internal val VdaNeutral22 = Color(0xFF38342F)
internal val VdaNeutral24 = Color(0xFF3C3933)
internal val VdaNeutral87 = Color(0xFFDED9D2)
internal val VdaNeutral90 = Color(0xFFE7E2DB)
internal val VdaNeutral92 = Color(0xFFECE7E0)
internal val VdaNeutral94 = Color(0xFFF2EDE6)
internal val VdaNeutral95 = Color(0xFFF5F0E9)
internal val VdaNeutral96 = Color(0xFFF8F3EC)
internal val VdaNeutral98 = Color(0xFFFEF9F2)
internal val VdaNeutral100 = Color(0xFFFFFFFF)

internal val VdaNeutralVariant30 = Color(0xFF4E4637)
internal val VdaNeutralVariant50 = Color(0xFF807666)
internal val VdaNeutralVariant60 = Color(0xFF9A8F7F)
internal val VdaNeutralVariant80 = Color(0xFFD0C5B4)
internal val VdaNeutralVariant90 = Color(0xFFEDE1D0)

/**
 * Extended colours - deliberately identical in light and dark. Every video preview sits on
 * [VdaVideoPlate] in both themes so the plate never influences how the content on it reads,
 * and so the grayscale depth output has no bright frame around it to read it against.
 */
internal val VdaVideoPlate = VdaNeutral4
internal val VdaOnVideoPlate = VdaNeutral90

/** Scrim alpha for chips/controls drawn over live video. 0.72 keeps text >= 4.5:1 even over a white frame. */
internal const val VdaVideoScrimAlpha = 0.72f
