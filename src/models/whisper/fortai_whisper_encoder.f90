module fortai_whisper_encoder
    !! Native Whisper encoder graph and execution.
    !!
    !! All model graph construction is in Fortran.  fortai_ggml is only the
    !! low-level tensor/backend ABI; it never selects a model or builds a
    !! Whisper graph on FortAI's behalf.
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_int, c_int64_t, c_loc, c_ptr, c_size_t, c_null_ptr
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32
    use fortai_ggml, only: GGML_STATUS_SUCCESS, GGML_TYPE_F16, GGML_TYPE_F32, &
        ggml_backend_sched_alloc_graph, ggml_backend_sched_buffer_size, ggml_backend_sched_graph_compute, &
        ggml_backend_sched_new, ggml_backend_sched_free, ggml_backend_sched_synchronize, ggml_backend_for_cpu, &
        ggml_backend_free, ggml_backend_set_tensor, &
        ggml_backend_get_tensor, ggml_context_free, ggml_context_init, ggml_graph_build, ggml_graph_new, &
        ggml_graph_overhead, ggml_init_params, ggml_tensor_add, ggml_tensor_cast, ggml_tensor_cont_2d, &
        ggml_tensor_conv_1d_ph, ggml_tensor_flash_attn_ext, ggml_tensor_gelu, ggml_tensor_mul, ggml_tensor_mul_mat, &
        ggml_tensor_new_2d, ggml_tensor_new_3d, ggml_tensor_norm, ggml_tensor_permute, ggml_tensor_reshape_2d, &
        ggml_tensor_reshape_3d, ggml_tensor_set_input, ggml_tensor_set_output, ggml_tensor_soft_max_ext
    use fortai_status, only: FORTAI_INVALID, FORTAI_OUT_OF_MEMORY, FORTAI_UNSUPPORTED, status_t
    use fortai_whisper_audio, only: whisper_mel_t
    use fortai_whisper_model, only: whisper_native_model_t
    implicit none
    private

    logical(c_bool), parameter :: WHISPER_C_TRUE = .true._c_bool
    logical(c_bool), parameter :: WHISPER_C_FALSE = .false._c_bool

    integer(int32), parameter :: WHISPER_GRAPH_NODES = 4096_int32
    real(real32), parameter :: WHISPER_EPS = 1.0e-5_real32

    type, public :: whisper_encoder_t
        type(whisper_native_model_t), pointer :: model => null()
        type(c_ptr) :: context = c_null_ptr
        type(c_ptr) :: graph = c_null_ptr
        type(c_ptr) :: scheduler = c_null_ptr
        type(c_ptr) :: fallback_backend = c_null_ptr
        type(c_ptr) :: mel_input = c_null_ptr
        type(c_ptr) :: output = c_null_ptr
        integer(int32) :: n_ctx = 0_int32
        integer(int32) :: n_state = 0_int32
        integer(int32) :: n_mels = 0_int32
        integer(int32) :: n_heads = 0_int32
        logical :: flash_attention = .true.
        logical :: ready = .false.
        integer(int64) :: compute_bytes = 0_int64
    contains
        procedure :: init => whisper_encoder_init
        procedure :: close => whisper_encoder_close
        procedure :: encode => whisper_encoder_encode
        procedure :: memory_bytes => whisper_encoder_memory_bytes
        procedure :: is_ready => whisper_encoder_is_ready
    end type whisper_encoder_t

contains

    subroutine whisper_encoder_init(self, model, flash_attention, stat)
        class(whisper_encoder_t), intent(inout) :: self
        class(whisper_native_model_t), target, intent(inout) :: model
        logical, intent(in), optional :: flash_attention
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: metadata_bytes
        integer(int32) :: layer
        logical :: use_flash
        type(c_ptr) :: cur, input, qcur, kcur, vcur, query, key, value, scores, probabilities, attended
        type(c_ptr) :: inp_l, inp_ff
        type(c_ptr) :: weight, bias
        character(len=128) :: name
        integer(c_int64_t) :: shape3(3)
        integer(c_int64_t) :: nctx, nstate, nhead, head_dim
        integer(c_int) :: code
        logical :: allocated_ok

        call stat%clear()
        call self%close()
        if (.not. model%is_ready()) then
            call stat%set(FORTAI_INVALID, 'Whisper encoder requires a loaded model')
            return
        end if
        self%model => model
        self%n_ctx = model%file%hparams%n_audio_ctx
        self%n_state = model%file%hparams%n_audio_state
        self%n_mels = model%file%hparams%n_mels
        self%n_heads = model%file%hparams%n_audio_head
        use_flash = .true.
        if (present(flash_attention)) use_flash = flash_attention
        self%flash_attention = use_flash

        nctx = int(self%n_ctx, c_int64_t)
        nstate = int(self%n_state, c_int64_t)
        nhead = int(self%n_heads, c_int64_t)
        head_dim = nstate / nhead

        ! The context stores tensor metadata only.  The scheduler owns the
        ! reusable device compute arena, so this remains small even for a
        ! 1500-frame large-v3 graph.
        metadata_bytes = max(16_c_size_t * 1024_c_size_t * 1024_c_size_t, &
            int(WHISPER_GRAPH_NODES, c_size_t) * (ggml_graph_overhead() + 4_c_size_t * 1024_c_size_t))
        self%context = ggml_context_init(ggml_init_params(metadata_bytes, WHISPER_C_TRUE))
        if (.not. c_associated(self%context)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper encoder graph metadata allocation failed')
            return
        end if

        self%mel_input = ggml_tensor_new_2d(self%context, GGML_TYPE_F32, 2_int64 * nctx, int(self%n_mels, c_int64_t))
        call ggml_tensor_set_input(self%mel_input)
        cur = ggml_tensor_conv_1d_ph(self%context, model%tensor_by_name('encoder.conv1.weight'), self%mel_input, 1_c_int, 1_c_int)
        cur = ggml_tensor_add(self%context, cur, model%tensor_by_name('encoder.conv1.bias'))
        cur = ggml_tensor_gelu(self%context, cur)
        cur = ggml_tensor_conv_1d_ph(self%context, model%tensor_by_name('encoder.conv2.weight'), cur, 2_c_int, 1_c_int)
        cur = ggml_tensor_add(self%context, cur, model%tensor_by_name('encoder.conv2.bias'))
        cur = ggml_tensor_gelu(self%context, cur)

        cur = ggml_tensor_add(self%context, model%tensor_by_name('encoder.positional_embedding'), &
            ggml_tensor_cont_2d(self%context, ggml_tensor_permute(self%context, cur, 1_c_int, 0_c_int, 2_c_int, 3_c_int), &
            nstate, nctx))
        inp_l = cur

        do layer = 0_int32, model%file%hparams%n_audio_layer - 1_int32
            call whisper_encoder_name(name, layer, 'attn_ln.weight')
            weight = model%tensor_by_name(trim(name))
            call whisper_encoder_name(name, layer, 'attn_ln.bias')
            bias = model%tensor_by_name(trim(name))
            cur = ggml_tensor_norm(self%context, inp_l, WHISPER_EPS)
            cur = ggml_tensor_add(self%context, ggml_tensor_mul(self%context, cur, weight), bias)

            call whisper_encoder_name(name, layer, 'attn.query.weight')
            qcur = ggml_tensor_mul_mat(self%context, model%tensor_by_name(trim(name)), cur)
            call whisper_encoder_name(name, layer, 'attn.query.bias')
            qcur = ggml_tensor_add(self%context, qcur, model%tensor_by_name(trim(name)))
            call whisper_encoder_name(name, layer, 'attn.key.weight')
            kcur = ggml_tensor_mul_mat(self%context, model%tensor_by_name(trim(name)), cur)
            call whisper_encoder_name(name, layer, 'attn.value.weight')
            vcur = ggml_tensor_mul_mat(self%context, model%tensor_by_name(trim(name)), cur)
            call whisper_encoder_name(name, layer, 'attn.value.bias')
            vcur = ggml_tensor_add(self%context, vcur, model%tensor_by_name(trim(name)))

            shape3(1) = head_dim
            shape3(2) = nhead
            shape3(3) = nctx
            query = ggml_tensor_permute(self%context, ggml_tensor_reshape_3d(self%context, qcur, shape3(1), shape3(2), shape3(3)), &
                0_c_int, 2_c_int, 1_c_int, 3_c_int)
            if (self%flash_attention) then
                key = ggml_tensor_permute(self%context, ggml_tensor_cast(self%context, &
                    ggml_tensor_reshape_3d(self%context, kcur, shape3(1), shape3(2), shape3(3)), GGML_TYPE_F16), &
                    0_c_int, 2_c_int, 1_c_int, 3_c_int)
                value = ggml_tensor_permute(self%context, ggml_tensor_cast(self%context, &
                    ggml_tensor_reshape_3d(self%context, vcur, shape3(1), shape3(2), shape3(3)), GGML_TYPE_F16), &
                    0_c_int, 2_c_int, 1_c_int, 3_c_int)
                attended = ggml_tensor_flash_attn_ext(self%context, query, key, value, c_null_ptr, &
                    1.0_real32 / sqrt(real(head_dim, real32)), 0.0_real32, 0.0_real32)
                cur = ggml_tensor_reshape_2d(self%context, attended, nstate, nctx)
            else
                key = ggml_tensor_permute(self%context, ggml_tensor_cast(self%context, &
                    ggml_tensor_reshape_3d(self%context, kcur, shape3(1), shape3(2), shape3(3)), GGML_TYPE_F16), &
                    0_c_int, 2_c_int, 1_c_int, 3_c_int)
                value = ggml_tensor_cast(self%context, ggml_tensor_permute(self%context, &
                    ggml_tensor_reshape_3d(self%context, vcur, shape3(1), shape3(2), shape3(3)), &
                    1_c_int, 2_c_int, 0_c_int, 3_c_int), GGML_TYPE_F16)
                scores = ggml_tensor_mul_mat(self%context, key, query)
                probabilities = ggml_tensor_soft_max_ext(self%context, scores, c_null_ptr, &
                    1.0_real32 / sqrt(real(head_dim, real32)), 0.0_real32)
                attended = ggml_tensor_mul_mat(self%context, value, probabilities)
                cur = ggml_tensor_cont_2d(self%context, ggml_tensor_permute(self%context, attended, &
                    0_c_int, 2_c_int, 1_c_int, 3_c_int), nstate, nctx)
            end if
            call whisper_encoder_name(name, layer, 'attn.out.weight')
            cur = ggml_tensor_mul_mat(self%context, model%tensor_by_name(trim(name)), cur)
            call whisper_encoder_name(name, layer, 'attn.out.bias')
            cur = ggml_tensor_add(self%context, cur, model%tensor_by_name(trim(name)))
            cur = ggml_tensor_add(self%context, cur, inp_l)
            inp_ff = cur

            call whisper_encoder_name(name, layer, 'mlp_ln.weight')
            weight = model%tensor_by_name(trim(name))
            call whisper_encoder_name(name, layer, 'mlp_ln.bias')
            bias = model%tensor_by_name(trim(name))
            cur = ggml_tensor_norm(self%context, inp_ff, WHISPER_EPS)
            cur = ggml_tensor_add(self%context, ggml_tensor_mul(self%context, cur, weight), bias)
            call whisper_encoder_name(name, layer, 'mlp.0.weight')
            cur = ggml_tensor_mul_mat(self%context, model%tensor_by_name(trim(name)), cur)
            call whisper_encoder_name(name, layer, 'mlp.0.bias')
            cur = ggml_tensor_add(self%context, cur, model%tensor_by_name(trim(name)))
            cur = ggml_tensor_gelu(self%context, cur)
            call whisper_encoder_name(name, layer, 'mlp.2.weight')
            cur = ggml_tensor_mul_mat(self%context, model%tensor_by_name(trim(name)), cur)
            call whisper_encoder_name(name, layer, 'mlp.2.bias')
            cur = ggml_tensor_add(self%context, cur, model%tensor_by_name(trim(name)))
            inp_l = ggml_tensor_add(self%context, cur, inp_ff)
        end do

        cur = ggml_tensor_norm(self%context, inp_l, WHISPER_EPS)
        cur = ggml_tensor_add(self%context, ggml_tensor_mul(self%context, cur, model%tensor_by_name('encoder.ln_post.weight')), &
            model%tensor_by_name('encoder.ln_post.bias'))
        self%output = cur
        call ggml_tensor_set_output(self%output)
        self%graph = ggml_graph_new(self%context, int(WHISPER_GRAPH_NODES, c_size_t), WHISPER_C_FALSE)
        call ggml_graph_build(self%graph, self%output)
        self%fallback_backend = ggml_backend_for_cpu()
        if (.not. c_associated(self%fallback_backend)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper encoder CPU fallback backend allocation failed')
            call self%close()
            return
        end if
        self%scheduler = ggml_backend_sched_new(model%backend, int(WHISPER_GRAPH_NODES, c_size_t), WHISPER_C_FALSE, &
            WHISPER_C_TRUE, self%fallback_backend)
        if (.not. c_associated(self%scheduler)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper encoder scheduler allocation failed')
            call self%close()
            return
        end if
        allocated_ok = ggml_backend_sched_alloc_graph(self%scheduler, self%graph)
        if (.not. allocated_ok) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper encoder compute arena allocation failed')
            call self%close()
            return
        end if
        self%compute_bytes = int(ggml_backend_sched_buffer_size(self%scheduler, model%backend), int64)
        self%ready = .true.
    end subroutine whisper_encoder_init

    subroutine whisper_encoder_name(name, layer, suffix)
        character(len=*), intent(out) :: name
        integer(int32), intent(in) :: layer
        character(len=*), intent(in) :: suffix

        write(name, '("encoder.blocks.",i0,".",a)') layer, trim(suffix)
    end subroutine whisper_encoder_name

    subroutine whisper_encoder_encode(self, mel, output, stat)
        class(whisper_encoder_t), intent(inout) :: self
        type(whisper_mel_t), intent(in) :: mel
        real(real32), allocatable, target, intent(out) :: output(:,:)
        type(status_t), intent(out) :: stat
        real(real32), allocatable, target :: packed(:)
        integer(int32) :: i, j, n_time
        integer(c_int) :: code
        integer(c_size_t) :: nbytes

        call stat%clear()
        if (.not. self%ready .or. .not. associated(self%model)) then
            call stat%set(FORTAI_INVALID, 'Whisper encoder is not initialized')
            return
        end if
        if (.not. allocated(mel%data) .or. mel%n_mel /= self%n_mels) then
            call stat%set(FORTAI_INVALID, 'Whisper mel shape does not match the encoder')
            return
        end if
        n_time = min(2_int32 * self%n_ctx, mel%n_len)
        allocate(packed(2 * self%n_ctx * self%n_mels))
        packed = 0.0_real32
        do j = 1, self%n_mels
            do i = 1, n_time
                ! GGML's [time, mel] tensor is time-contiguous per channel;
                ! mel_t is deliberately Fortran-friendly [mel,time].
                packed((j - 1) * 2 * self%n_ctx + i) = mel%data(j, i)
            end do
        end do
        nbytes = int(size(packed), c_size_t) * int(storage_size(packed(1)) / 8, c_size_t)
        call ggml_backend_set_tensor(self%mel_input, c_loc(packed(1)), 0_c_size_t, nbytes)
        code = ggml_backend_sched_graph_compute(self%scheduler, self%graph)
        if (code /= GGML_STATUS_SUCCESS) then
            deallocate(packed)
            call stat%set(FORTAI_UNSUPPORTED, 'Whisper encoder graph execution failed')
            return
        end if
        call ggml_backend_sched_synchronize(self%scheduler)
        allocate(output(self%n_state, self%n_ctx))
        nbytes = int(size(output), c_size_t) * int(storage_size(output(1, 1)) / 8, c_size_t)
        call ggml_backend_get_tensor(self%output, c_loc(output(1, 1)), 0_c_size_t, nbytes)
        deallocate(packed)
    end subroutine whisper_encoder_encode

    subroutine whisper_encoder_close(self)
        class(whisper_encoder_t), intent(inout) :: self

        call ggml_backend_sched_free(self%scheduler)
        call ggml_backend_free(self%fallback_backend)
        call ggml_context_free(self%context)
        self%graph = c_null_ptr
        self%mel_input = c_null_ptr
        self%output = c_null_ptr
        self%fallback_backend = c_null_ptr
        nullify(self%model)
        self%n_ctx = 0_int32
        self%n_state = 0_int32
        self%n_mels = 0_int32
        self%n_heads = 0_int32
        self%compute_bytes = 0_int64
        self%ready = .false.
    end subroutine whisper_encoder_close

    integer(int64) function whisper_encoder_memory_bytes(self)
        class(whisper_encoder_t), intent(in) :: self

        whisper_encoder_memory_bytes = self%compute_bytes
    end function whisper_encoder_memory_bytes

    logical function whisper_encoder_is_ready(self)
        class(whisper_encoder_t), intent(in) :: self

        whisper_encoder_is_ready = self%ready
    end function whisper_encoder_is_ready

end module fortai_whisper_encoder
