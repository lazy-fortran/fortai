module fortai_qwen35
    use, intrinsic :: iso_fortran_env, only: int32
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    type, public :: qwen35_config_t
        integer(int32) :: hidden_size = 0_int32
        integer(int32) :: layer_count = 0_int32
        integer(int32) :: attention_layer_count = 0_int32
        integer(int32) :: recurrent_layer_count = 0_int32
    contains
        procedure :: init => qwen35_config_init
        procedure :: validate => qwen35_config_validate
    end type qwen35_config_t

contains

    subroutine qwen35_config_init(self, hidden_size, layer_count, &
            attention_layer_count, recurrent_layer_count, stat)
        class(qwen35_config_t), intent(inout) :: self
        integer(int32), intent(in) :: hidden_size
        integer(int32), intent(in) :: layer_count
        integer(int32), intent(in) :: attention_layer_count
        integer(int32), intent(in) :: recurrent_layer_count
        type(status_t), intent(out), optional :: stat

        self%hidden_size = hidden_size
        self%layer_count = layer_count
        self%attention_layer_count = attention_layer_count
        self%recurrent_layer_count = recurrent_layer_count
        if (present(stat)) call self%validate(stat)
    end subroutine qwen35_config_init

    subroutine qwen35_config_validate(self, stat)
        class(qwen35_config_t), intent(in) :: self
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (self%hidden_size <= 0_int32) then
            call stat%set(FORTAI_INVALID, 'Qwen hidden size must be positive')
            return
        end if
        if (self%layer_count <= 0_int32) then
            call stat%set(FORTAI_INVALID, 'Qwen layer count must be positive')
            return
        end if
        if (self%attention_layer_count < 0_int32) then
            call stat%set(FORTAI_INVALID, 'attention layer count cannot be negative')
            return
        end if
        if (self%recurrent_layer_count < 0_int32) then
            call stat%set(FORTAI_INVALID, 'recurrent layer count cannot be negative')
            return
        end if
        if (self%attention_layer_count + self%recurrent_layer_count &
            /= self%layer_count) then
            call stat%set(FORTAI_INVALID, 'Qwen layer counts do not add up')
        end if
    end subroutine qwen35_config_validate

end module fortai_qwen35
