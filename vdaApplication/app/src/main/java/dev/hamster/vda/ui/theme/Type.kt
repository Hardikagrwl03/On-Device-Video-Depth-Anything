package dev.hamster.vda.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

private val Base = Typography()

internal val VdaTypography = Base.copy(
    displayLarge = Base.displayLarge.copy(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Light,
        fontSize = 64.sp,
        letterSpacing = 10.sp,
        lineHeight = 72.sp
    ),
    titleLarge = Base.titleLarge.copy(
        fontWeight = FontWeight.SemiBold,
        letterSpacing = 0.sp
    ),
    labelLarge = Base.labelLarge.copy(
        fontWeight = FontWeight.Medium,
        letterSpacing = 0.4.sp
    ),
    bodySmall = Base.bodySmall.copy(
        lineHeight = 18.sp
    )
)
