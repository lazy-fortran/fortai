module fortai_whisper_format
    !! Native reader for the legacy GGML Whisper model container.
    !!
    !! Whisper weights are not GGUF: the file contains hparams, mel filters,
    !! vocabulary, and a stream of named tensors.  Keeping this reader separate
    !! from fortai_gguf_runtime lets the model family own its format while the
    !! backend remains format independent.
    use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32
    use fortai_status, only: FORTAI_INVALID, FORTAI_IO_ERROR, FORTAI_OUT_OF_MEMORY, FORTAI_UNSUPPORTED, status_t
    use fortai_string, only: string_t
    implicit none
    private

    integer(int32), parameter, public :: WHISPER_MAGIC = int(z'67676d6c', int32)
    integer(int32), parameter, public :: WHISPER_SAMPLE_RATE = 16000_int32
    integer(int32), parameter, public :: WHISPER_N_FFT = 400_int32
    integer(int32), parameter, public :: WHISPER_HOP_LENGTH = 160_int32
    integer(int32), parameter, public :: WHISPER_CHUNK_LENGTH = 30_int32

    type, public :: whisper_hparams_t
        integer(int32) :: n_vocab = 0_int32
        integer(int32) :: n_audio_ctx = 0_int32
        integer(int32) :: n_audio_state = 0_int32
        integer(int32) :: n_audio_head = 0_int32
        integer(int32) :: n_audio_layer = 0_int32
        integer(int32) :: n_text_ctx = 0_int32
        integer(int32) :: n_text_state = 0_int32
        integer(int32) :: n_text_head = 0_int32
        integer(int32) :: n_text_layer = 0_int32
        integer(int32) :: n_mels = 0_int32
        integer(int32) :: ftype = 0_int32
    contains
        procedure :: validate => whisper_hparams_validate
        procedure :: is_large_v3_turbo => whisper_hparams_is_large_v3_turbo
    end type whisper_hparams_t

    type, public :: whisper_vocab_t
        type(string_t), allocatable :: token(:)
    contains
        procedure :: clear => whisper_vocab_clear
        procedure :: count => whisper_vocab_count
    end type whisper_vocab_t

    type, public :: whisper_tensor_record_t
        type(string_t) :: name
        integer(int32) :: rank = 0_int32
        integer(int32) :: value_type = -1_int32
        integer(int64) :: shape(4) = 1_int64
        integer(int64) :: elements = 0_int64
        integer(int64) :: byte_count = 0_int64
        integer(int64) :: file_offset = 0_int64
    end type whisper_tensor_record_t

    type, public :: whisper_file_t
        type(whisper_hparams_t) :: hparams
        integer(int32) :: filter_mel = 0_int32
        integer(int32) :: filter_fft = 0_int32
        real(real32), allocatable :: filters(:,:)
        type(whisper_vocab_t) :: vocab
        type(whisper_tensor_record_t), allocatable :: tensors(:)
        character(len=:), allocatable :: path
        integer(int64) :: file_size = 0_int64
        integer :: unit = -1
        logical :: opened = .false.
    contains
        procedure :: open => whisper_file_open
        procedure :: close => whisper_file_close
        procedure :: tensor_index => whisper_file_tensor_index
        procedure :: tensor_bytes => whisper_file_tensor_bytes
        procedure :: read_tensor => whisper_file_read_tensor
        procedure :: validate => whisper_file_validate
    end type whisper_file_t

    public :: whisper_ggml_type_size
    public :: whisper_ggml_block_size

contains

    subroutine whisper_hparams_validate(self, stat)
        class(whisper_hparams_t), intent(in) :: self
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (self%n_vocab <= 0 .or. self%n_audio_ctx <= 0 .or. self%n_audio_state <= 0 .or. &
            self%n_audio_head <= 0 .or. self%n_audio_layer <= 0 .or. self%n_text_ctx <= 0 .or. &
            self%n_text_state <= 0 .or. self%n_text_head <= 0 .or. self%n_text_layer <= 0 .or. &
            self%n_mels <= 0) then
            call stat%set(FORTAI_INVALID, 'Whisper hparams contain a non-positive dimension')
            return
        end if
        if (mod(self%n_audio_state, self%n_audio_head) /= 0 .or. &
            mod(self%n_text_state, self%n_text_head) /= 0) then
            call stat%set(FORTAI_INVALID, 'Whisper state dimensions are not divisible by head counts')
            return
        end if
        if (self%n_audio_state /= self%n_text_state) then
            call stat%set(FORTAI_INVALID, 'Whisper encoder and decoder state sizes differ')
            return
        end if
        if (self%ftype < 0) call stat%set(FORTAI_INVALID, 'Whisper weight type is invalid')
    end subroutine whisper_hparams_validate

    logical function whisper_hparams_is_large_v3_turbo(self)
        class(whisper_hparams_t), intent(in) :: self

        whisper_hparams_is_large_v3_turbo = self%n_vocab == 51866 .and. self%n_audio_ctx == 1500 .and. &
            self%n_audio_state == 1280 .and. self%n_audio_head == 20 .and. self%n_audio_layer == 32 .and. &
            self%n_text_ctx == 448 .and. self%n_text_state == 1280 .and. self%n_text_head == 20 .and. &
            self%n_text_layer == 4 .and. self%n_mels == 128
    end function whisper_hparams_is_large_v3_turbo

    subroutine whisper_vocab_clear(self)
        class(whisper_vocab_t), intent(inout) :: self

        if (allocated(self%token)) deallocate(self%token)
    end subroutine whisper_vocab_clear

    integer(int32) function whisper_vocab_count(self)
        class(whisper_vocab_t), intent(in) :: self

        if (allocated(self%token)) then
            whisper_vocab_count = int(size(self%token), int32)
        else
            whisper_vocab_count = 0_int32
        end if
    end function whisper_vocab_count

    integer(int64) function whisper_ggml_block_size(value_type)
        integer(int32), intent(in) :: value_type

        select case (value_type)
        case (0, 1, 24, 25, 26, 27, 28, 30, 39, 40)
            whisper_ggml_block_size = 1_int64
        case (2, 3, 6, 7, 8, 9)
            whisper_ggml_block_size = 32_int64
        case (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 34, 35, 41, 42)
            whisper_ggml_block_size = 256_int64
        case default
            whisper_ggml_block_size = 0_int64
        end select
    end function whisper_ggml_block_size

    integer(int64) function whisper_ggml_type_size(value_type)
        integer(int32), intent(in) :: value_type

        select case (value_type)
        case (0)
            whisper_ggml_type_size = 4_int64
        case (1)
            whisper_ggml_type_size = 2_int64
        case (2)
            whisper_ggml_type_size = 18_int64
        case (3)
            whisper_ggml_type_size = 20_int64
        case (6)
            whisper_ggml_type_size = 22_int64
        case (7)
            whisper_ggml_type_size = 24_int64
        case (8)
            whisper_ggml_type_size = 34_int64
        case (9)
            whisper_ggml_type_size = 36_int64
        case (10)
            whisper_ggml_type_size = 84_int64
        case (11)
            whisper_ggml_type_size = 110_int64
        case (12)
            whisper_ggml_type_size = 144_int64
        case (13)
            whisper_ggml_type_size = 176_int64
        case (14)
            whisper_ggml_type_size = 210_int64
        case (15)
            whisper_ggml_type_size = 292_int64
        case (16)
            whisper_ggml_type_size = 66_int64
        case (17)
            whisper_ggml_type_size = 74_int64
        case (18)
            whisper_ggml_type_size = 98_int64
        case (19)
            whisper_ggml_type_size = 56_int64
        case (20)
            whisper_ggml_type_size = 36_int64
        case (21)
            whisper_ggml_type_size = 110_int64
        case (22)
            whisper_ggml_type_size = 82_int64
        case (23)
            whisper_ggml_type_size = 136_int64
        case (24)
            whisper_ggml_type_size = 1_int64
        case (25)
            whisper_ggml_type_size = 2_int64
        case (26)
            whisper_ggml_type_size = 4_int64
        case (27)
            whisper_ggml_type_size = 8_int64
        case (28)
            whisper_ggml_type_size = 8_int64
        case (29)
            whisper_ggml_type_size = 56_int64
        case (30)
            whisper_ggml_type_size = 2_int64
        case (34)
            whisper_ggml_type_size = 20_int64
        case (35)
            whisper_ggml_type_size = 66_int64
        case (39)
            whisper_ggml_type_size = 1_int64
        case (40)
            whisper_ggml_type_size = 18_int64
        case (41)
            whisper_ggml_type_size = 18_int64
        case (42)
            whisper_ggml_type_size = 66_int64
        case default
            whisper_ggml_type_size = 0_int64
        end select
    end function whisper_ggml_type_size

    subroutine whisper_file_close(self)
        class(whisper_file_t), intent(inout) :: self

        if (self%opened) close(self%unit)
        call self%vocab%clear()
        if (allocated(self%filters)) deallocate(self%filters)
        if (allocated(self%tensors)) deallocate(self%tensors)
        if (allocated(self%path)) deallocate(self%path)
        self%hparams = whisper_hparams_t()
        self%filter_mel = 0_int32
        self%filter_fft = 0_int32
        self%file_size = 0_int64
        self%unit = -1
        self%opened = .false.
    end subroutine whisper_file_close

    subroutine whisper_file_open(self, path, stat)
        class(whisper_file_t), intent(inout) :: self
        character(len=*), intent(in) :: path
        type(status_t), intent(out) :: stat
        integer :: unit, ios
        integer(int32) :: magic, n_vocab, length, n_dims, value_type
        integer(int32) :: i, j, dimension32
        integer(int64) :: position, next_position, elements, bytes
        integer(int64) :: dimensions(4)
        character(len=:), allocatable :: token_text, tensor_name
        type(whisper_tensor_record_t), allocatable :: records(:), grown(:)
        integer :: record_count
        type(whisper_hparams_t) :: hparams
        real(real32), allocatable :: filter_values(:)
        integer(int32) :: filter_mel, filter_fft
        logical :: header_pending

        call stat%clear()
        call self%close()
        if (len_trim(path) == 0) then
            call stat%set(FORTAI_INVALID, 'Whisper model path is empty')
            return
        end if

        open(newunit=unit, file=trim(path), access='stream', form='unformatted', status='old', action='read', &
            convert='little_endian', iostat=ios)
        if (ios /= 0) then
            call stat%set(FORTAI_IO_ERROR, 'Unable to open Whisper model: ' // trim(path))
            return
        end if

        inquire(unit=unit, size=self%file_size)
        read(unit, iostat=ios) magic
        if (ios /= 0 .or. magic /= WHISPER_MAGIC) then
            close(unit)
            call stat%set(FORTAI_INVALID, 'Whisper model has invalid GGML magic')
            return
        end if

        read(unit, iostat=ios) hparams%n_vocab, hparams%n_audio_ctx, hparams%n_audio_state, &
            hparams%n_audio_head, hparams%n_audio_layer, hparams%n_text_ctx, hparams%n_text_state, &
            hparams%n_text_head, hparams%n_text_layer, hparams%n_mels, hparams%ftype
        if (ios /= 0) then
            close(unit)
            call stat%set(FORTAI_IO_ERROR, 'Whisper model hparams are truncated')
            return
        end if
        call hparams%validate(stat)
        if (.not. stat%is_ok()) then
            close(unit)
            return
        end if

        read(unit, iostat=ios) filter_mel, filter_fft
        if (ios /= 0 .or. filter_mel <= 0 .or. filter_fft <= 0) then
            close(unit)
            call stat%set(FORTAI_INVALID, 'Whisper mel filter metadata is invalid')
            return
        end if
        allocate(filter_values(filter_mel * filter_fft))
        read(unit, iostat=ios) filter_values
        if (ios /= 0) then
            deallocate(filter_values)
            close(unit)
            call stat%set(FORTAI_IO_ERROR, 'Whisper mel filter data is truncated')
            return
        end if

        read(unit, iostat=ios) n_vocab
        if (ios /= 0 .or. n_vocab < 0 .or. n_vocab > hparams%n_vocab) then
            deallocate(filter_values)
            close(unit)
            call stat%set(FORTAI_INVALID, 'Whisper vocabulary count is invalid')
            return
        end if
        call self%vocab%clear()
        allocate(self%vocab%token(n_vocab))
        do i = 1, n_vocab
            read(unit, iostat=ios) length
            if (ios /= 0 .or. length < 0 .or. length > 1024 * 1024) then
                deallocate(filter_values)
                close(unit)
                call self%close()
                call stat%set(FORTAI_INVALID, 'Whisper vocabulary token length is invalid')
                return
            end if
            allocate(character(len=length) :: token_text)
            if (length > 0) read(unit, iostat=ios) token_text
            if (ios /= 0) then
                deallocate(token_text, filter_values)
                close(unit)
                call self%close()
                call stat%set(FORTAI_IO_ERROR, 'Whisper vocabulary is truncated')
                return
            end if
            call self%vocab%token(i)%set(token_text)
            deallocate(token_text)
        end do

        record_count = 0
        allocate(records(0))
        header_pending = .false.
        do
            if (.not. header_pending) then
                read(unit, iostat=ios) n_dims
                if (ios < 0) exit
                if (ios /= 0) then
                    deallocate(records, filter_values)
                    close(unit)
                    call self%close()
                    call stat%set(FORTAI_IO_ERROR, 'Whisper tensor index is truncated')
                    return
                end if
            end if
            header_pending = .false.
            read(unit, iostat=ios) length, value_type
            if (ios /= 0 .or. n_dims < 1 .or. n_dims > 4 .or. length < 0 .or. length > 4096) then
                deallocate(records, filter_values)
                close(unit)
                call self%close()
                call stat%set(FORTAI_INVALID, 'Whisper tensor header is invalid')
                return
            end if
            dimensions = 1_int64
            elements = 1_int64
            do j = 1, n_dims
                read(unit, iostat=ios) dimension32
                dimensions(j) = int(dimension32, int64)
                if (ios /= 0 .or. dimension32 <= 0) then
                    deallocate(records, filter_values)
                    close(unit)
                    call self%close()
                    call stat%set(FORTAI_INVALID, 'Whisper tensor shape is invalid')
                    return
                end if
                elements = elements * dimensions(j)
            end do
            allocate(character(len=length) :: tensor_name)
            if (length > 0) read(unit, iostat=ios) tensor_name
            if (ios /= 0) then
                deallocate(tensor_name, records, filter_values)
                close(unit)
                call self%close()
                call stat%set(FORTAI_IO_ERROR, 'Whisper tensor name is truncated')
                return
            end if
            bytes = whisper_ggml_tensor_bytes(value_type, elements)
            if (bytes <= 0) then
                deallocate(tensor_name, records, filter_values)
                close(unit)
                call self%close()
                call stat%set(FORTAI_UNSUPPORTED, 'Whisper tensor type is unsupported')
                return
            end if
            inquire(unit=unit, pos=position)
            if (position <= 0 .or. position + bytes - 1 > self%file_size) then
                deallocate(tensor_name, records, filter_values)
                close(unit)
                call self%close()
                call stat%set(FORTAI_IO_ERROR, 'Whisper tensor payload exceeds the model file')
                return
            end if
            record_count = record_count + 1
            allocate(grown(record_count))
            if (record_count > 1) grown(:record_count - 1) = records
            call grown(record_count)%name%set(tensor_name)
            grown(record_count)%rank = n_dims
            grown(record_count)%value_type = value_type
            grown(record_count)%shape = dimensions
            grown(record_count)%elements = elements
            grown(record_count)%byte_count = bytes
            grown(record_count)%file_offset = position
            call move_alloc(grown, records)
            deallocate(tensor_name)
            next_position = position + bytes
            if (next_position > self%file_size) exit
            read(unit, pos=next_position, iostat=ios) n_dims
            if (ios < 0) exit
            if (ios /= 0) then
                deallocate(records, filter_values)
                close(unit)
                call self%close()
                call stat%set(FORTAI_IO_ERROR, 'Whisper tensor index is truncated')
                return
            end if
            header_pending = .true.
        end do
        self%unit = unit

        allocate(self%filters(filter_mel, filter_fft))
        do j = 1, filter_fft
            do i = 1, filter_mel
                ! The legacy GGML file stores each mel row contiguously
                ! ([mel][fft]); the Fortran array is [mel,fft].  Index the
                ! file row explicitly instead of treating it as a Fortran
                ! column-major matrix.
                self%filters(i, j) = filter_values((i - 1) * filter_fft + j)
            end do
        end do
        deallocate(filter_values)
        self%hparams = hparams
        self%filter_mel = filter_mel
        self%filter_fft = filter_fft
        self%tensors = records
        self%path = trim(path)
        self%opened = .true.
        call self%validate(stat)
        if (.not. stat%is_ok()) call self%close()
    end subroutine whisper_file_open

    integer(int64) function whisper_ggml_tensor_bytes(value_type, elements)
        integer(int32), intent(in) :: value_type
        integer(int64), intent(in) :: elements
        integer(int64) :: block_size, type_size

        block_size = whisper_ggml_block_size(value_type)
        type_size = whisper_ggml_type_size(value_type)
        if (block_size <= 0 .or. type_size <= 0) then
            whisper_ggml_tensor_bytes = 0_int64
        else if (mod(elements, block_size) /= 0) then
            whisper_ggml_tensor_bytes = 0_int64
        else
            whisper_ggml_tensor_bytes = (elements / block_size) * type_size
        end if
    end function whisper_ggml_tensor_bytes

    subroutine whisper_file_validate(self, stat)
        class(whisper_file_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer :: i
        integer(int64) :: expected

        call stat%clear()
        if (.not. self%opened) then
            call stat%set(FORTAI_INVALID, 'Whisper model is not open')
            return
        end if
        call self%hparams%validate(stat)
        if (.not. stat%is_ok()) return
        if (self%filter_mel /= self%hparams%n_mels .or. self%filter_fft /= 1 + WHISPER_N_FFT / 2) then
            call stat%set(FORTAI_INVALID, 'Whisper mel filter dimensions do not match hparams')
            return
        end if
        if (.not. allocated(self%filters)) then
            call stat%set(FORTAI_INVALID, 'Whisper mel filters are missing')
            return
        end if
        if (size(self%filters, 1) /= self%filter_mel .or. size(self%filters, 2) /= self%filter_fft) then
            call stat%set(FORTAI_INVALID, 'Whisper mel filters are missing')
            return
        end if
        if (self%vocab%count() <= 0) then
            call stat%set(FORTAI_INVALID, 'Whisper vocabulary is empty')
            return
        end if
        if (.not. allocated(self%tensors)) then
            call stat%set(FORTAI_INVALID, 'Whisper model contains no tensors')
            return
        end if
        if (size(self%tensors) == 0) then
            call stat%set(FORTAI_INVALID, 'Whisper model contains no tensors')
            return
        end if
        do i = 1, size(self%tensors)
            expected = whisper_ggml_tensor_bytes(self%tensors(i)%value_type, self%tensors(i)%elements)
            if (expected /= self%tensors(i)%byte_count .or. self%tensors(i)%file_offset <= 0) then
                call stat%set(FORTAI_INVALID, 'Whisper tensor byte accounting is inconsistent')
                return
            end if
        end do
    end subroutine whisper_file_validate

    integer(int32) function whisper_file_tensor_index(self, name)
        class(whisper_file_t), intent(in) :: self
        character(len=*), intent(in) :: name
        integer :: i

        whisper_file_tensor_index = 0_int32
        if (.not. allocated(self%tensors)) return
        do i = 1, size(self%tensors)
            if (self%tensors(i)%name%equals(name)) then
                whisper_file_tensor_index = int(i, int32)
                return
            end if
        end do
    end function whisper_file_tensor_index

    integer(int64) function whisper_file_tensor_bytes(self, index, stat)
        class(whisper_file_t), intent(in) :: self
        integer(int32), intent(in) :: index
        type(status_t), intent(out), optional :: stat

        if (present(stat)) call stat%clear()
        whisper_file_tensor_bytes = 0_int64
        if (.not. allocated(self%tensors)) then
            if (present(stat)) call stat%set(FORTAI_INVALID, 'Whisper tensor index is out of range')
            return
        end if
        if (index < 1 .or. index > size(self%tensors)) then
            if (present(stat)) call stat%set(FORTAI_INVALID, 'Whisper tensor index is out of range')
            return
        end if
        whisper_file_tensor_bytes = self%tensors(index)%byte_count
    end function whisper_file_tensor_bytes

    subroutine whisper_file_read_tensor(self, index, bytes, stat)
        class(whisper_file_t), intent(in) :: self
        integer(int32), intent(in) :: index
        integer(int8), allocatable, intent(out) :: bytes(:)
        type(status_t), intent(out) :: stat
        integer :: ios
        integer(int64) :: length
        character(len=32) :: ios_text

        call stat%clear()
        if (allocated(bytes)) deallocate(bytes)
        if (.not. self%opened) then
            allocate(bytes(0))
            call stat%set(FORTAI_INVALID, 'Whisper tensor index is out of range')
            return
        end if
        if (.not. allocated(self%tensors)) then
            allocate(bytes(0))
            call stat%set(FORTAI_INVALID, 'Whisper tensor index is out of range')
            return
        end if
        if (index < 1 .or. index > size(self%tensors)) then
            allocate(bytes(0))
            call stat%set(FORTAI_INVALID, 'Whisper tensor index is out of range')
            return
        end if
        length = self%tensors(index)%byte_count
        if (length > int(huge(0), int64)) then
            allocate(bytes(0))
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper tensor is too large for host indexing')
            return
        end if
        allocate(bytes(int(length)))
        if (.not. self%opened) then
            ios = 1
        else
            read(self%unit, pos=self%tensors(index)%file_offset, iostat=ios) bytes
        end if
        if (ios /= 0) then
            deallocate(bytes)
            allocate(bytes(0))
            write (ios_text, '(i0)') ios
            call stat%set(FORTAI_IO_ERROR, 'Unable to read Whisper tensor payload (iostat=' // &
                trim(ios_text) // '): ' // self%tensors(index)%name%as_character())
        end if
    end subroutine whisper_file_read_tensor

end module fortai_whisper_format
