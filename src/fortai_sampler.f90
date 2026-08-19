module fortai_sampler
    use, intrinsic :: iso_fortran_env, only: int32, real64
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    type, public :: sampler_t
        real(real64) :: temperature = 0.0_real64
        integer(int32) :: top_k = 1_int32
    contains
        procedure :: greedy => sampler_greedy
    end type sampler_t

contains

    subroutine sampler_greedy(self, logits, token_id, stat)
        class(sampler_t), intent(in) :: self
        real(real64), intent(in) :: logits(:)
        integer(int32), intent(out) :: token_id
        type(status_t), intent(out) :: stat
        integer(int32) :: i
        real(real64) :: best

        call stat%clear()
        token_id = 0_int32
        if (size(logits) == 0) then
            call stat%set(FORTAI_INVALID, 'cannot sample an empty logit vector')
            return
        end if

        token_id = 1_int32
        best = logits(1)
        do i = 2, size(logits)
            if (logits(i) > best) then
                best = logits(i)
                token_id = i
            end if
        end do
    end subroutine sampler_greedy

end module fortai_sampler
