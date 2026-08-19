module fortai_qwen35_dflash2
    implicit none
    private

    public :: qwen35_dflash2_available

contains

    logical function qwen35_dflash2_available()
        qwen35_dflash2_available = .false.
    end function qwen35_dflash2_available

end module fortai_qwen35_dflash2
