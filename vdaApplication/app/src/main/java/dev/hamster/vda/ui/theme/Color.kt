package dev.hamster.vda.ui.theme

import androidx.compose.ui.graphics.Color

// Brand tonal ramps. Hue 203.5 (deep cyan-teal) for primary/secondary/neutrals,
// hue 62 (warm amber) for tertiary. Tone == CIE L*. Generated once and verified for
// WCAG contrast; see docs/ui-redesign-plan.md "Key decision 3". Do not hand-edit
// individual values - regenerate the whole ramp if the hue ever changes.
internal val VdaPrimary10 = Color(0xFF002022)
internal val VdaPrimary20 = Color(0xFF003739)
internal val VdaPrimary30 = Color(0xFF004F53)
internal val VdaPrimary40 = Color(0xFF00696E)
internal val VdaPrimary80 = Color(0xFF61D7DF)
internal val VdaPrimary90 = Color(0xFFB6ECEF)
internal val VdaPrimary100 = Color(0xFFFFFFFF)

internal val VdaSecondary10 = Color(0xFF002022)
internal val VdaSecondary20 = Color(0xFF153537)
internal val VdaSecondary30 = Color(0xFF2C4C4E)
internal val VdaSecondary40 = Color(0xFF446466)
internal val VdaSecondary80 = Color(0xFFAACDCF)
internal val VdaSecondary90 = Color(0xFFC6E9EB)
internal val VdaSecondary100 = Color(0xFFFFFFFF)

internal val VdaTertiary10 = Color(0xFF2C1700)
internal val VdaTertiary20 = Color(0xFF4D2600)
internal val VdaTertiary30 = Color(0xFF6F3802)
internal val VdaTertiary40 = Color(0xFF8C4F1B)
internal val VdaTertiary80 = Color(0xFFF5BA8F)
internal val VdaTertiary90 = Color(0xFFFEDCC5)
internal val VdaTertiary100 = Color(0xFFFFFFFF)

internal val VdaNeutral0 = Color(0xFF000000)
internal val VdaNeutral4 = Color(0xFF090F10)
internal val VdaNeutral6 = Color(0xFF0F1414)
internal val VdaNeutral10 = Color(0xFF181C1C)
internal val VdaNeutral12 = Color(0xFF1C2021)
internal val VdaNeutral17 = Color(0xFF262B2B)
internal val VdaNeutral20 = Color(0xFF2D3131)
internal val VdaNeutral22 = Color(0xFF313636)
internal val VdaNeutral24 = Color(0xFF363A3A)
internal val VdaNeutral87 = Color(0xFFD5DBDB)
internal val VdaNeutral90 = Color(0xFFDEE3E4)
internal val VdaNeutral92 = Color(0xFFE4E9E9)
internal val VdaNeutral94 = Color(0xFFE9EFEF)
internal val VdaNeutral95 = Color(0xFFECF2F2)
internal val VdaNeutral96 = Color(0xFFEFF5F5)
internal val VdaNeutral98 = Color(0xFFF5FAFB)
internal val VdaNeutral100 = Color(0xFFFFFFFF)

internal val VdaNeutralVariant30 = Color(0xFF3F4849)
internal val VdaNeutralVariant50 = Color(0xFF6F7979)
internal val VdaNeutralVariant60 = Color(0xFF889393)
internal val VdaNeutralVariant80 = Color(0xFFBDC9C9)
internal val VdaNeutralVariant90 = Color(0xFFD9E5E5)

/**
 * Extended colours - deliberately identical in light and dark. Every video preview sits on
 * [VdaVideoPlate] in both themes so the plate never influences how the content on it reads,
 * and so the grayscale depth output has no bright frame around it to read it against.
 */
internal val VdaVideoPlate = VdaNeutral4
internal val VdaOnVideoPlate = VdaNeutral90

/** Scrim alpha for chips/controls drawn over live video. 0.72 keeps text >= 4.5:1 even over a white frame. */
internal const val VdaVideoScrimAlpha = 0.72f
