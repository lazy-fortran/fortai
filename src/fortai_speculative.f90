module fortai_speculative
    use, intrinsic :: iso_fortran_env, only: int32
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    type, public :: speculative_t
        integer(int32) :: max_accepted = 8_int32
    contains
        procedure :: accept_prefix => speculative_accept_prefix
    end type speculative_t

contains

    subroutine speculative_accept_prefix(self, target_ids, draft_ids, accepted, stat)
        class(speculative_t), intent(in) :: self
        integer(int32), intent(in) :: target_ids(:)
        integer(int32), intent(in) :: draft_ids(:)
        integer(int32), intent(out) :: accepted
        type(status_t), intent(out) :: stat
        integer(int32) :: i, limit

        call stat%clear()
        accepted = 0_int32
        if (self%max_accepted <= 0_int32) then
            call stat%set(FORTAI_INVALID, 'maximum accepted tokens must be positive')
            return
        end if
        limit = min(size(target_ids), size(draft_ids))
        limit = min(limit, self%max_accepted)
        do i = 1, limit
            if (target_ids(i) /= draft_ids(i)) return
            accepted = accepted + 1_int32
        end do
    end subroutine speculative_accept_prefix

end module fortai_speculative
