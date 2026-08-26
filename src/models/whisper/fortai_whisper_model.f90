module fortai_whisper_model
    !! Native Whisper weight ownership and backend placement.
    !!
    !! This layer knows the Whisper file's tensor names and dimensions but does
    !! not implement inference.  Graph semantics stay in Fortran model code;
    !! fortai_ggml only supplies the low-level tensor/backend ABI.
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_int, c_int64_t, c_loc, c_ptr, c_size_t, c_null_ptr
    use, intrinsic :: iso_fortran_env, only: int8, int32, int64
    use fortai_ggml, only: GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_GPU, ggml_backend_alloc_context, &
        ggml_backend_buffer_size, ggml_backend_for_device, ggml_backend_for_cpu, ggml_backend_free, &
        ggml_backend_free_buffer, ggml_backend_set_tensor, ggml_backend_synchronize, ggml_backend_device, &
        ggml_context_free, ggml_context_init, ggml_init_params, ggml_tensor_new, ggml_tensor_nbytes, &
        ggml_tensor_overhead
    use fortai_status, only: FORTAI_INVALID, FORTAI_OUT_OF_MEMORY, FORTAI_UNSUPPORTED, status_t
    use fortai_whisper_format, only: whisper_file_t
    implicit none
    private

    logical(c_bool), parameter :: WHISPER_C_TRUE = .true._c_bool

    type, public :: whisper_native_model_t
        type(whisper_file_t) :: file
        type(c_ptr) :: device = c_null_ptr
        type(c_ptr) :: backend = c_null_ptr
        type(c_ptr) :: weight_context = c_null_ptr
        type(c_ptr) :: weight_buffer = c_null_ptr
        type(c_ptr), allocatable :: tensor(:)
        integer(int64) :: weight_bytes = 0_int64
        logical :: use_gpu = .false.
        integer(int32) :: gpu_device = 0_int32
        logical :: ready = .false.
    contains
        procedure :: open => whisper_native_model_open
        procedure :: close => whisper_native_model_close
        procedure :: tensor_by_name => whisper_native_model_tensor
        procedure :: synchronize => whisper_native_model_synchronize
        procedure :: memory_bytes => whisper_native_model_memory_bytes
        procedure :: is_ready => whisper_native_model_is_ready
    end type whisper_native_model_t

    public :: whisper_native_model_load_tensor

contains

    subroutine whisper_native_model_open(self, path, use_gpu, gpu_device, stat)
        class(whisper_native_model_t), intent(inout) :: self
        character(len=*), intent(in) :: path
        logical, intent(in), optional :: use_gpu
        integer(int32), intent(in), optional :: gpu_device
        type(status_t), intent(out) :: stat
        type(whisper_native_model_t) :: candidate
        integer :: i
        integer(int32) :: ordinal
        integer(c_size_t) :: metadata_bytes
        integer(c_int64_t) :: shape(4)
        integer(c_int) :: value_type

        call stat%clear()
        call self%close()
        candidate%use_gpu = .false.
        if (present(use_gpu)) candidate%use_gpu = use_gpu
        ordinal = 0_int32
        if (present(gpu_device)) ordinal = max(0_int32, gpu_device)
        candidate%gpu_device = ordinal

        call candidate%file%open(path, stat)
        if (.not. stat%is_ok()) return
        if (.not. candidate%file%hparams%is_large_v3_turbo()) then
            call candidate%file%close()
            call stat%set(FORTAI_UNSUPPORTED, 'Only Whisper large-v3-turbo GGML models are supported')
            return
        end if
        metadata_bytes = max(1_c_size_t, int(size(candidate%file%tensors), c_size_t) * &
            max(ggml_tensor_overhead(), 1024_c_size_t) + 1024_c_size_t * 1024_c_size_t)

        candidate%device = ggml_backend_device(merge(GGML_BACKEND_DEVICE_TYPE_GPU, &
            GGML_BACKEND_DEVICE_TYPE_CPU, candidate%use_gpu), ordinal)
        if (.not. c_associated(candidate%device)) then
            call candidate%file%close()
            call stat%set(FORTAI_UNSUPPORTED, 'Requested Whisper backend device is unavailable')
            return
        end if
        if (candidate%use_gpu) then
            candidate%backend = ggml_backend_for_device(candidate%device)
        else
            candidate%backend = ggml_backend_for_cpu()
        end if
        if (.not. c_associated(candidate%backend)) then
            call candidate%file%close()
            call stat%set(FORTAI_UNSUPPORTED, 'Requested Whisper backend could not be initialized')
            return
        end if

        candidate%weight_context = ggml_context_init(ggml_init_params(metadata_bytes, WHISPER_C_TRUE))
        if (.not. c_associated(candidate%weight_context)) then
            call ggml_backend_free(candidate%backend)
            call candidate%file%close()
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper weight metadata allocation failed')
            return
        end if
        allocate(candidate%tensor(size(candidate%file%tensors)))
        candidate%tensor = c_null_ptr
        do i = 1, size(candidate%file%tensors)
            shape = candidate%file%tensors(i)%shape
            value_type = int(candidate%file%tensors(i)%value_type, c_int)
            candidate%tensor(i) = ggml_tensor_new(candidate%weight_context, value_type, &
                shape(1:candidate%file%tensors(i)%rank))
            if (.not. c_associated(candidate%tensor(i))) then
                call candidate%close()
                call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper weight tensor metadata allocation failed')
                return
            end if
        end do
        candidate%weight_buffer = ggml_backend_alloc_context(candidate%weight_context, candidate%backend)
        if (.not. c_associated(candidate%weight_buffer)) then
            call candidate%close()
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper weight buffer allocation failed')
            return
        end if
        candidate%weight_bytes = int(ggml_backend_buffer_size(candidate%weight_buffer), int64)

        do i = 1, size(candidate%tensor)
            call whisper_native_model_load_tensor(candidate, i, stat)
            if (.not. stat%is_ok()) then
                call candidate%close()
                return
            end if
        end do
        call candidate%synchronize()
        candidate%ready = .true.
        call move_whisper_model(candidate, self)
    end subroutine whisper_native_model_open

    subroutine move_whisper_model(source, destination)
        type(whisper_native_model_t), intent(inout) :: source
        class(whisper_native_model_t), intent(inout) :: destination

        call destination%close()
        call move_alloc(source%file%tensors, destination%file%tensors)
        call move_alloc(source%file%filters, destination%file%filters)
        call move_alloc(source%file%vocab%token, destination%file%vocab%token)
        if (allocated(source%file%path)) call move_alloc(source%file%path, destination%file%path)
        destination%file%hparams = source%file%hparams
        destination%file%filter_mel = source%file%filter_mel
        destination%file%filter_fft = source%file%filter_fft
        destination%file%file_size = source%file%file_size
        destination%file%unit = source%file%unit
        destination%file%opened = source%file%opened
        source%file%unit = -1
        source%file%opened = .false.
        destination%device = source%device
        destination%backend = source%backend
        destination%weight_context = source%weight_context
        destination%weight_buffer = source%weight_buffer
        call move_alloc(source%tensor, destination%tensor)
        destination%weight_bytes = source%weight_bytes
        destination%use_gpu = source%use_gpu
        destination%gpu_device = source%gpu_device
        destination%ready = source%ready
        source%device = c_null_ptr
        source%backend = c_null_ptr
        source%weight_context = c_null_ptr
        source%weight_buffer = c_null_ptr
        source%weight_bytes = 0_int64
        source%ready = .false.
    end subroutine move_whisper_model

    subroutine whisper_native_model_load_tensor(self, index, stat)
        class(whisper_native_model_t), intent(inout) :: self
        integer, intent(in) :: index
        type(status_t), intent(out) :: stat
        integer(int8), allocatable, target :: bytes(:)
        integer(c_size_t) :: expected

        call stat%clear()
        if (.not. allocated(self%tensor)) then
            call stat%set(FORTAI_INVALID, 'Whisper weight tensor index is invalid')
            return
        end if
        if (index < 1 .or. index > size(self%tensor)) then
            call stat%set(FORTAI_INVALID, 'Whisper weight tensor index is invalid')
            return
        end if
        call self%file%read_tensor(int(index, int32), bytes, stat)
        if (.not. stat%is_ok()) return
        expected = ggml_tensor_nbytes(self%tensor(index))
        if (int(size(bytes), c_size_t) /= expected) then
            deallocate(bytes)
            call stat%set(FORTAI_INVALID, 'Whisper weight tensor byte count does not match metadata: ' // &
                self%file%tensors(index)%name%as_character())
            return
        end if
        if (expected > 0_c_size_t) call ggml_backend_set_tensor(self%tensor(index), c_loc(bytes(1)), 0_c_size_t, expected)
        deallocate(bytes)
    end subroutine whisper_native_model_load_tensor

    subroutine whisper_native_model_close(self)
        class(whisper_native_model_t), intent(inout) :: self

        call ggml_backend_free_buffer(self%weight_buffer)
        call ggml_context_free(self%weight_context)
        call ggml_backend_free(self%backend)
        if (allocated(self%tensor)) deallocate(self%tensor)
        call self%file%close()
        self%device = c_null_ptr
        self%weight_bytes = 0_int64
        self%use_gpu = .false.
        self%gpu_device = 0_int32
        self%ready = .false.
    end subroutine whisper_native_model_close

    function whisper_native_model_tensor(self, name, stat) result(tensor)
        class(whisper_native_model_t), intent(in) :: self
        character(len=*), intent(in) :: name
        type(status_t), intent(out), optional :: stat
        type(c_ptr) :: tensor
        integer(int32) :: index

        if (present(stat)) call stat%clear()
        tensor = c_null_ptr
        index = self%file%tensor_index(name)
        if (index <= 0 .or. .not. allocated(self%tensor)) then
            if (present(stat)) call stat%set(FORTAI_INVALID, 'Whisper tensor is missing: ' // trim(name))
            return
        end if
        tensor = self%tensor(index)
    end function whisper_native_model_tensor

    subroutine whisper_native_model_synchronize(self)
        class(whisper_native_model_t), intent(inout) :: self

        if (c_associated(self%backend)) call ggml_backend_synchronize(self%backend)
    end subroutine whisper_native_model_synchronize

    integer(int64) function whisper_native_model_memory_bytes(self)
        class(whisper_native_model_t), intent(in) :: self

        whisper_native_model_memory_bytes = self%weight_bytes
    end function whisper_native_model_memory_bytes

    logical function whisper_native_model_is_ready(self)
        class(whisper_native_model_t), intent(in) :: self

        whisper_native_model_is_ready = self%ready
    end function whisper_native_model_is_ready

end module fortai_whisper_model
