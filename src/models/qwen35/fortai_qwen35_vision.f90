module fortai_qwen35_vision
    implicit none
    private

    public :: qwen35_vision_available

contains

    logical function qwen35_vision_available()
        qwen35_vision_available = .false.
    end function qwen35_vision_available

end module fortai_qwen35_vision
