from enum import Enum

class LedPosition(Enum):
    # All
    ALL = 'AllLeds'
    # Head
    BRAIN = 'BrainLeds'
    BRAIN_BACK = 'BrainLedsBack'
    BRAIN_MIDDLE = 'BrainLedsMiddle'
    BRAIN_FRONT = 'BrainLedsFront'
    BRAIN_LEFT = 'BrainLedsLeft'
    BRAIN_RIGHT = 'BrainLedsRight'
    # Ears
    EAR = 'EarLeds'
    RIGHT_EAR = 'RightEarLeds'
    LEFT_EAR = 'LeftEarLeds'
    RIGHT_EAR_BACK = 'RightEarLedsBack'
    RIGHT_EAR_FRONT = 'RightEarLedsFront'
    LEFT_EAR_BACK = 'LeftEarLedsBack'
    LEFT_EAR_FRONT = 'LeftEarLedsFront'
    RIGHT_EAR_EVEN = 'RightEarLedsEven'
    RIGHT_EAR_ODD = 'RightEarLedsOdd'
    LEFT_EAR_EVEN = 'LeftEarLedsEven'
    LEFT_EAR_ODD = 'LeftEarLedsOdd'
    # Face
    FACE = 'FaceLeds'
    RIGHT_FACE = 'RightFaceLeds'
    LEFT_FACE = 'LeftFaceLeds'
    FACE_BOTTOM = 'FaceLedsBottom'
    FACE_EXTERNAL = 'FaceLedsExternal'
    FACE_INTERNAL = 'FaceLedsInternal'
    FACE_TOP = 'FaceLedsTop'
    FACE_RIGHT_BOTTOM = 'FaceLedsRightBottom'
    FACE_RIGHT_EXTERNAL = 'FaceLedsRightExternal'
    FACE_RIGHT_INTERNAL = 'FaceLedsRightInternal'
    FACE_RIGHT_TOP = 'FaceLedsRightTop'
    FACE_LEFT_BOTTOM = 'FaceLedsLeftBottom'
    FACE_LEFT_EXTERNAL = 'FaceLedsLeftExternal'
    FACE_LEFT_INTERNAL = 'FaceLedsLeftInternal'
    FACE_LEFT_TOP = 'FaceLedsLeftTop'
    # Chest
    CHEST = 'ChestLeds'
    # Feet
    FEET = 'FeetLeds'
    LEFT_FOOT = 'LeftFootLeds'
    RIGHT_FOOT = 'RightFootLeds'


class Postures(Enum):
    CROUCH = 'Crouch'
    LYING_BACK = 'LyingBack'
    LYING_BELLY = 'LyingBelly'
    SIT = 'Sit'
    SIT_RELAX = 'SitRelax'
    STAND = 'Stand'
    STAND_INIT = 'StandInit'
    STAND_ZERO = 'StandZero'
