module fortai_backend_cpu
    use, intrinsic :: iso_fortran_env, only: real64
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    public :: cpu_matvec

contains

    subroutine cpu_matvec(matrix, vector, result, stat)
        real(real64), intent(in) :: matrix(:,:)
        real(real64), intent(in) :: vector(:)
        real(real64), allocatable, intent(out) :: result(:)
        type(status_t), intent(out) :: stat
        integer :: i, j

        call stat%clear()
        allocate (result(size(matrix, 1)))
        result = 0.0_real64
        if (size(matrix, 2) /= size(vector)) then
            call stat%set(FORTAI_INVALID, 'matvec dimensions do not agree')
            return
        end if

        do i = 1, size(matrix, 1)
            do j = 1, size(matrix, 2)
                result(i) = result(i) + matrix(i, j) * vector(j)
            end do
        end do
    end subroutine cpu_matvec

end module fortai_backend_cpu
