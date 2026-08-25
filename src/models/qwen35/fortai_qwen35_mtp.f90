module fortai_qwen35_mtp
    implicit none
    private

    public :: qwen35_mtp_available

contains

    logical function qwen35_mtp_available()
        ! The native Qwen3.8 NextN/MTP graph is implemented in
        ! fortai_qwen35_cpu.  Model-level availability is still reported by
        ! qwen35_cpu_model_t%mtp_available because ordinary GGUF files do not
        ! carry the optional block.
        qwen35_mtp_available = .true.
    end function qwen35_mtp_available

end module fortai_qwen35_mtp
