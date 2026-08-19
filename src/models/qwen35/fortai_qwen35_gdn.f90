module fortai_qwen35_gdn
    use, intrinsic :: iso_fortran_env, only: real64
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    public :: gdn_reference_step

contains

    subroutine gdn_reference_step(state, input, decay, gate, output, stat)
        real(real64), intent(inout) :: state(:)
        real(real64), intent(in) :: input(:)
        real(real64), intent(in) :: decay
        real(real64), intent(in) :: gate(:)
        real(real64), allocatable, intent(out) :: output(:)
        type(status_t), intent(out) :: stat
        integer :: i

        call stat%clear()
        allocate (output(size(state)))
        output = 0.0_real64
        if (size(input) /= size(state)) then
            call stat%set(FORTAI_INVALID, 'GDN input and state sizes differ')
            return
        end if
        if (size(gate) /= size(state)) then
            call stat%set(FORTAI_INVALID, 'GDN gate and state sizes differ')
            return
        end if

        do i = 1, size(state)
            state(i) = decay * state(i) + input(i)
            output(i) = gate(i) * state(i)
        end do
    end subroutine gdn_reference_step

end module fortai_qwen35_gdn
