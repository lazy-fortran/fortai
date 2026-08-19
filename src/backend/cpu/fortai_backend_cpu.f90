module fortai_backend_cpu
    use, intrinsic :: iso_fortran_env, only: real64
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    public :: cpu_matvec
    public :: cpu_matvec_inplace

contains

    subroutine cpu_matvec(matrix, vector, result, stat)
        real(real64), intent(in) :: matrix(:,:)
        real(real64), intent(in) :: vector(:)
        real(real64), allocatable, intent(out) :: result(:)
        type(status_t), intent(out) :: stat

        allocate (result(size(matrix, 1)))
        call cpu_matvec_inplace(matrix, vector, result, stat)
    end subroutine cpu_matvec

    subroutine cpu_matvec_inplace(matrix, vector, result, stat)
        real(real64), intent(in) :: matrix(:,:)
        real(real64), intent(in) :: vector(:)
        real(real64), intent(inout) :: result(:)
        type(status_t), intent(out) :: stat
        integer :: i, j
        real(real64) :: accumulator

        call stat%clear()
        if (size(matrix, 2) /= size(vector)) then
            call stat%set(FORTAI_INVALID, 'matvec dimensions do not agree')
            return
        end if
        if (size(result) /= size(matrix, 1)) then
            call stat%set(FORTAI_INVALID, 'matvec output has the wrong shape')
            return
        end if

        !$omp parallel do default(none) shared(matrix, vector, result) &
        !$omp& private(i, j, accumulator) schedule(static)
        do i = 1, size(matrix, 1)
            accumulator = 0.0_real64
            !$omp simd reduction(+:accumulator)
            do j = 1, size(matrix, 2)
                accumulator = accumulator + matrix(i, j) * vector(j)
            end do
            result(i) = accumulator
        end do
        !$omp end parallel do
    end subroutine cpu_matvec_inplace

end module fortai_backend_cpu
