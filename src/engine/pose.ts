// Approximate head pose from 478-pt landmarks, used only to rank a person's source images
// against a target so the closest-pose shot supplies the texture. No MediaPipe transformation
// matrices, no model re-config — cheap geometry is accurate enough for nearest-neighbour picks.

import { IDX, type Landmarks } from './types'

interface Pose {
  yaw: number
  pitch: number
  roll: number
}

type P2 = [number, number]

export function estimatePose(l: Landmarks): Pose {
  const p = l.points
  const g = (i: number): P2 => [p[i * 2], p[i * 2 + 1]]
  const rIris = g(IDX.rightIris),
    lIris = g(IDX.leftIris)
  const nose = g(IDX.noseTip),
    chin = g(IDX.chin)
  const rFace = g(IDX.rightFace),
    lFace = g(IDX.leftFace)

  // roll: tilt of the iris line (radians; 0 = level head)
  const roll = Math.atan2(lIris[1] - rIris[1], lIris[0] - rIris[0])

  // roll-corrected axes: U along the eye line, V perpendicular (down). Measuring yaw/pitch on
  // raw x/y axes breaks under head tilt, so project onto these instead.
  const c = Math.cos(roll),
    s = Math.sin(roll)
  const U = (a: P2, b: P2) => (a[0] - b[0]) * c + (a[1] - b[1]) * s
  const V = (a: P2, b: P2) => -(a[0] - b[0]) * s + (a[1] - b[1]) * c

  // yaw: asymmetry of the nose tip between the two cheek edges, in [-1, 1]
  const dR = Math.abs(U(nose, rFace))
  const dL = Math.abs(U(lFace, nose))
  const yaw = (dL - dR) / (dL + dR + 1e-6)

  // pitch: eye-line->nose vs nose->chin perpendicular spans, ~[-1, 1]
  const eyeMid: P2 = [(rIris[0] + lIris[0]) / 2, (rIris[1] + lIris[1]) / 2]
  const upper = V(nose, eyeMid)
  const lower = V(chin, nose)
  const pitch = (upper - lower) / (Math.abs(upper) + Math.abs(lower) + 1e-6)

  return { yaw, pitch, roll }
}

export function poseDistance(a: Pose, b: Pose): number {
  const dy = a.yaw - b.yaw,
    dp = a.pitch - b.pitch,
    dr = a.roll - b.roll
  // Yaw drives texture self-occlusion, so weight it fully. Roll is corrected by the mesh->mesh
  // warp itself (pure in-plane rotation), so weight it low; it only mildly affects lighting.
  return Math.sqrt(dy * dy + dp * dp + 0.25 * dr * dr)
}
