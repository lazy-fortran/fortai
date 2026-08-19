module fortai_qwen35_mtp
    implicit none
    private

    public :: qwen35_mtp_available

contains

    logical function qwen35_mtp_available()
        qwen35_mtp_available = .false.
    end function qwen35_mtp_available

end module fortai_qwen35_mtp
