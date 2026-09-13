package dev.hamster.vda.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

/**
 * Corner scale for surfaces (cards, sheets, preview frames). Buttons deliberately do NOT read
 * from this: `ButtonDefaults.shape` resolves to `ShapeKeyTokens.CornerFull` -> `CircleShape`
 * independently of `MaterialTheme.shapes`, so buttons stay pills regardless of this scale.
 */
internal val VdaShapes = Shapes(
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(20.dp),
    extraLarge = RoundedCornerShape(28.dp)
)
