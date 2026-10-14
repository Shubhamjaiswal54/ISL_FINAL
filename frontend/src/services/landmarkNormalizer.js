/**
 * landmarkNormalizer.js
 * Client-side wrist-relative, scale-normalized coordinate normalization
 *
 * Normalizes MediaPipe hand landmarks by subtracting the wrist position and
 * dividing by hand extent, converting absolute coordinates to relative,
 * scale-invariant coordinates for model input. Must match
 * scripts/extract_kaggle_dataset.py and scripts/1_collect_landmarks.py
 * exactly, since training data and live inference both flow through this
 * same encoding.
 */

export const normalizeLandmarks = (rawLandmarks, handLabel = 'Right') => {
  // rawLandmarks: array of 21 {x, y, z} objects from MediaPipe

  if (!rawLandmarks || rawLandmarks.length === 0) {
    return null
  }

  // Find wrist (landmark 0)
  const wrist = rawLandmarks[0]

  // Subtract wrist from all landmarks
  const centered = rawLandmarks.map((lm, i) => {
    if (i === 0) return [0, 0, 0]  // Wrist is the reference, always 0 relative
    return [lm.x - wrist.x, lm.y - wrist.y, (lm.z || 0) - (wrist.z || 0)]
  })

  const scale = Math.max(
    ...centered.map(([x, y, z]) => Math.hypot(x, y, z)),
    1e-6,
  )
  // 21 landmarks x (x, y, z) = 63 values per hand
  const singleHand = centered.flatMap(([x, y, z]) => [x / scale, y / scale, z / scale])
  const emptyHand = new Array(63).fill(0)

  // Match extract_kaggle_dataset.py: left hand first, right hand second.
  return handLabel === 'Left'
    ? [...singleHand, ...emptyHand]
    : [...emptyHand, ...singleHand]
}

/**
 * Extract 21 hand landmarks from MediaPipe detection results
 * @param {Object} results - MediaPipe Hands detection results
 * @returns {Array} Array of 21 {x, y, z} landmark positions
 */
export const extractLandmarks = (results) => {
  if (!results.multi_hand_landmarks || results.multi_hand_landmarks.length === 0) {
    return null
  }

  // Take the first hand
  const handLandmarks = results.multi_hand_landmarks[0]
  const landmarks = []

  for (let i = 0; i < handLandmarks.length; i++) {
    const lm = handLandmarks[i]
    landmarks.push({
      x: lm.x,
      y: lm.y,
      z: lm.z || 0
    })
  }

  return landmarks
}