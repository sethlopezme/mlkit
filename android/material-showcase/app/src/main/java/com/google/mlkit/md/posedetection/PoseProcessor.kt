package com.google.mlkit.md.posedetection

import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.md.InputInfo
import com.google.mlkit.md.camera.FrameProcessorBase
import com.google.mlkit.md.camera.GraphicOverlay
import com.google.mlkit.md.camera.WorkflowModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.io.IOException
import kotlin.math.abs
import kotlin.math.atan2

class PoseProcessor(graphicOverlay: GraphicOverlay, private val workflowModel: WorkflowModel) :
    FrameProcessorBase<Pose>() {

    private val detector: PoseDetector

    companion object {
        private const val TAG = "PoseProcessor"
    }

    init {
        val options = PoseDetectorOptions.Builder()
            .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
            .build()
        detector = PoseDetection.getClient(options)
    }

    override fun detectInImage(image: InputImage): Task<Pose> = detector.process(image)

    override fun onSuccess(inputInfo: InputInfo, results: Pose, graphicOverlay: GraphicOverlay) {
        val isValidPose = isValidPose(results)
        if (!isValidPose) {
            workflowModel.setWorkflowState(WorkflowModel.WorkflowState.DETECTING)
        } else {
            workflowModel.setWorkflowState(WorkflowModel.WorkflowState.DETECTED)

            val isPowerPose = isPowerPose(results)
            if (isPowerPose) {
                workflowModel.setWorkflowState(WorkflowModel.WorkflowState.CONFIRMED)
            }
        }
    }

    override fun onFailure(e: Exception) {
        Log.e(TAG, "Pose detection failed!", e)
    }

    override fun stop() {
        super.stop()
        try {
            detector.close()
        } catch (e: IOException) {
            Log.e(TAG, "Failed to close pose detector!", e)
        }
    }

    private fun isValidPose(pose: Pose): Boolean {
        return pose.allPoseLandmarks.isNotEmpty()
    }

    private fun getAngle(startPoint: PoseLandmark, midPoint: PoseLandmark, endPoint: PoseLandmark): Double {
        var result = Math.toDegrees(
            (atan2(
                endPoint.getPosition().y - midPoint.getPosition().y,
                endPoint.getPosition().x - midPoint.getPosition().x
            )
                    - atan2(
                startPoint.getPosition().y - midPoint.getPosition().y,
                startPoint.getPosition().x - midPoint.getPosition().x
            )).toDouble()
        )
        result = abs(result) // Angle should never be negative
        if (result > 180) {
            result = 360.0 - result // Always get the acute representation of the angle
        }
        return result
    }

    private fun isPowerPose(pose: Pose): Boolean {
        val leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)!!
        val rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)!!
        val leftElbow = pose.getPoseLandmark(PoseLandmark.LEFT_ELBOW)!!
        val rightElbow = pose.getPoseLandmark(PoseLandmark.RIGHT_ELBOW)!!
        val leftWrist = pose.getPoseLandmark(PoseLandmark.LEFT_WRIST)!!
        val rightWrist = pose.getPoseLandmark(PoseLandmark.RIGHT_WRIST)!!
        val leftHip = pose.getPoseLandmark(PoseLandmark.LEFT_HIP)!!
        val rightHip = pose.getPoseLandmark(PoseLandmark.RIGHT_HIP)!!

        // ensure all landmarks are in the frame
        val landmarks =
            listOf(leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip)
        val hasAllLandmarksInFrame = landmarks.all { it.inFrameLikelihood >= 0.9f }
        if (!hasAllLandmarksInFrame) return false

        // determine positioning of landmarks...
        // 1. arms are below shoulders
        val leftElbowTooHigh = leftElbow.position.y <= leftShoulder.position.y
        val rightElbowTooHigh = rightElbow.position.y <= rightShoulder.position.y
        if (leftElbowTooHigh || rightElbowTooHigh) return false

        // 2. hands are within a certain range of the hips
        // ...

        // 3. arms are at the correct angle
        val leftArmAngle = getAngle(startPoint = leftShoulder, midPoint = leftElbow, endPoint = leftWrist)
        val rightArmAngle = getAngle(startPoint = rightShoulder, midPoint = rightElbow, endPoint = rightWrist)
        val leftArmOutOfRange = leftArmAngle > 90 || leftArmAngle < 70
        val rightArmOutOfRange = rightArmAngle > 90 || rightArmAngle < 70
        if (leftArmOutOfRange || rightArmOutOfRange) return false

        return true
    }
}
