module fortai_model_ir
    use, intrinsic :: iso_fortran_env, only: int32
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    type, public :: model_ir_t
        character(len=:), allocatable :: architecture
        character(len=:), allocatable :: quantization
        integer(int32) :: vocab_size = 0_int32
        integer(int32) :: hidden_size = 0_int32
        integer(int32) :: layer_count = 0_int32
        integer(int32) :: context_length = 0_int32
    contains
        procedure :: init => model_ir_init
        procedure :: validate => model_ir_validate
    end type model_ir_t

contains

    subroutine model_ir_init(self, architecture, vocab_size, hidden_size, &
            layer_count, context_length, quantization, stat)
        class(model_ir_t), intent(inout) :: self
        character(len=*), intent(in) :: architecture
        integer(int32), intent(in) :: vocab_size
        integer(int32), intent(in) :: hidden_size
        integer(int32), intent(in) :: layer_count
        integer(int32), intent(in) :: context_length
        character(len=*), intent(in) :: quantization
        type(status_t), intent(out), optional :: stat

        if (present(stat)) call stat%clear()
        self%architecture = architecture
        self%quantization = quantization
        self%vocab_size = vocab_size
        self%hidden_size = hidden_size
        self%layer_count = layer_count
        self%context_length = context_length
        if (present(stat)) call self%validate(stat)
    end subroutine model_ir_init

    subroutine model_ir_validate(self, stat)
        class(model_ir_t), intent(in) :: self
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (.not. allocated(self%architecture)) then
            call stat%set(FORTAI_INVALID, 'model architecture is missing')
            return
        end if
        if (len_trim(self%architecture) == 0) then
            call stat%set(FORTAI_INVALID, 'model architecture is empty')
            return
        end if
        if (.not. allocated(self%quantization)) then
            call stat%set(FORTAI_INVALID, 'model quantization is missing')
            return
        end if
        if (self%vocab_size <= 0_int32) then
            call stat%set(FORTAI_INVALID, 'model vocabulary must be positive')
            return
        end if
        if (self%hidden_size <= 0_int32) then
            call stat%set(FORTAI_INVALID, 'model hidden size must be positive')
            return
        end if
        if (self%layer_count <= 0_int32) then
            call stat%set(FORTAI_INVALID, 'model layer count must be positive')
            return
        end if
        if (self%context_length <= 0_int32) then
            call stat%set(FORTAI_INVALID, 'model context length must be positive')
        end if
    end subroutine model_ir_validate

end module fortai_model_ir
