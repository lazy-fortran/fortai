module fortai_whisper_decoder
    !! Native Whisper decoder state, cross-attention cache, and one-token step.
    !!
    !! The graph is rebuilt for the requested KV slot because the legacy GGML
    !! ABI encodes a view offset at graph-construction time.  The device
    !! scheduler is retained and reuses its compute arena between steps; only
    !! small metadata is rebuilt.  This keeps state and model semantics in
    !! Fortran while preserving the reference cache layout.
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_int, c_int64_t, c_loc, c_ptr, c_size_t, c_null_ptr
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32
    use fortai_ggml, only: GGML_STATUS_SUCCESS, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I32, &
        ggml_backend_alloc_context, ggml_backend_buffer_clear, ggml_backend_for_cpu, ggml_backend_free, &
        ggml_backend_free_buffer, ggml_backend_get_tensor, ggml_backend_sched_alloc_graph, &
        ggml_backend_sched_graph_compute, ggml_backend_sched_new, ggml_backend_sched_reset, &
        ggml_backend_sched_free, ggml_backend_sched_set_tensor_backend, &
        ggml_backend_set_tensor, &
        ggml_context_free, &
        ggml_context_init, ggml_context_reset, ggml_graph_build, ggml_graph_expand, ggml_graph_new, ggml_graph_overhead, &
        ggml_init_params, ggml_tensor_add, ggml_tensor_cast, ggml_tensor_cpy, ggml_tensor_flash_attn_ext, &
        ggml_tensor_gelu, ggml_tensor_get_rows, ggml_tensor_mul, ggml_tensor_mul_mat, ggml_tensor_new_1d, &
        ggml_tensor_new_3d, ggml_tensor_norm, ggml_tensor_permute, ggml_tensor_reshape_3d, &
        ggml_tensor_scale, ggml_tensor_set_input, ggml_tensor_set_output, ggml_tensor_transpose, ggml_tensor_view_1d, &
        ggml_tensor_view, ggml_tensor_view_2d, ggml_tensor_view_3d, &
        ggml_tensor_reshape_2d, ggml_tensor_cont_2d, ggml_tensor_soft_max_ext, ggml_tensor_nbytes, ggml_tensor_overhead
    use fortai_status, only: FORTAI_INVALID, FORTAI_OUT_OF_MEMORY, FORTAI_UNSUPPORTED, status_t
    use fortai_whisper_model, only: whisper_native_model_t
    implicit none
    private

    logical(c_bool), parameter :: WHISPER_C_TRUE = .true._c_bool
    logical(c_bool), parameter :: WHISPER_C_FALSE = .false._c_bool

    integer(int32), parameter :: WHISPER_GRAPH_NODES = 4096_int32
    real(real32), parameter :: WHISPER_EPS = 1.0e-5_real32

    type, public :: whisper_decoder_t
        type(whisper_native_model_t), pointer :: model => null()
        type(c_ptr) :: state_context = c_null_ptr
        type(c_ptr) :: state_buffer = c_null_ptr
        type(c_ptr) :: cross_k = c_null_ptr
        type(c_ptr) :: cross_v = c_null_ptr
        type(c_ptr) :: self_k = c_null_ptr
        type(c_ptr) :: self_v = c_null_ptr
        type(c_ptr) :: decode_context = c_null_ptr
        type(c_ptr) :: scheduler = c_null_ptr
        type(c_ptr) :: fallback_backend = c_null_ptr
        integer(int32) :: n_audio_ctx = 0_int32
        integer(int32) :: n_audio_ctx_pad = 0_int32
        integer(int32) :: n_text_ctx = 0_int32
        integer(int32) :: n_text_ctx_pad = 0_int32
        integer(int32) :: n_state = 0_int32
        integer(int32) :: n_heads = 0_int32
        integer(int32) :: n_layers = 0_int32
        logical :: flash_attention = .true.
        logical :: cross_ready = .false.
        integer(int64) :: cache_bytes = 0_int64
        integer(int64) :: compute_bytes = 0_int64
    contains
        procedure :: init => whisper_decoder_init
        procedure :: prepare_cross => whisper_decoder_prepare_cross
        procedure :: decode => whisper_decoder_decode
        procedure :: decode_tokens => whisper_decoder_decode_tokens
        procedure :: close => whisper_decoder_close
        procedure :: memory_bytes => whisper_decoder_memory_bytes
        procedure :: is_ready => whisper_decoder_is_ready
    end type whisper_decoder_t

contains

    subroutine whisper_decoder_init(self, model, flash_attention, stat)
        class(whisper_decoder_t), intent(inout) :: self
        class(whisper_native_model_t), target, intent(inout) :: model
        logical, intent(in), optional :: flash_attention
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: metadata_bytes
        integer(c_int64_t) :: head_dim, nctx_pad, nstate, nheads, nlayers
        logical :: use_flash

        call stat%clear()
        call self%close()
        if (.not. model%is_ready()) then
            call stat%set(FORTAI_INVALID, 'Whisper decoder requires a loaded model')
            return
        end if
        self%model => model
        self%n_audio_ctx = model%file%hparams%n_audio_ctx
        self%n_audio_ctx_pad = whisper_pad_256(self%n_audio_ctx)
        self%n_text_ctx = model%file%hparams%n_text_ctx
        self%n_text_ctx_pad = whisper_pad_256(self%n_text_ctx)
        self%n_state = model%file%hparams%n_text_state
        self%n_heads = model%file%hparams%n_text_head
        self%n_layers = model%file%hparams%n_text_layer
        use_flash = .true.
        if (present(flash_attention)) use_flash = flash_attention
        self%flash_attention = use_flash

        nstate = int(self%n_state, c_int64_t)
        nheads = int(self%n_heads, c_int64_t)
        nlayers = int(self%n_layers, c_int64_t)
        head_dim = nstate / nheads
        nctx_pad = int(self%n_text_ctx_pad, c_int64_t)
        metadata_bytes = max(4_c_size_t * 1024_c_size_t, 8_c_size_t * ggml_tensor_overhead())
        self%state_context = ggml_context_init(ggml_init_params(metadata_bytes, WHISPER_C_TRUE))
        if (.not. c_associated(self%state_context)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper decoder cache metadata allocation failed')
            return
        end if
        ! The reference cache is a flat [state, position, layer] stream.
        ! Attention views then choose the head/time strides appropriate to K
        ! and V; using a 3-D [head_dim,time,head] allocation here would make
        ! those views semantically (and numerically) wrong.
        self%cross_k = ggml_tensor_new_1d(self%state_context, GGML_TYPE_F16, &
            nstate * int(self%n_audio_ctx_pad, c_int64_t) * nlayers)
        self%cross_v = ggml_tensor_new_1d(self%state_context, GGML_TYPE_F16, &
            nstate * int(self%n_audio_ctx_pad, c_int64_t) * nlayers)
        self%self_k = ggml_tensor_new_1d(self%state_context, GGML_TYPE_F16, nstate * nctx_pad * nlayers)
        self%self_v = ggml_tensor_new_1d(self%state_context, GGML_TYPE_F16, nstate * nctx_pad * nlayers)
        self%state_buffer = ggml_backend_alloc_context(self%state_context, model%backend)
        if (.not. c_associated(self%state_buffer)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper decoder cache allocation failed')
            call self%close()
            return
        end if
        call ggml_backend_buffer_clear(self%state_buffer, 0)
        self%cache_bytes = int(ggml_tensor_nbytes(self%cross_k) + ggml_tensor_nbytes(self%cross_v) + &
            ggml_tensor_nbytes(self%self_k) + ggml_tensor_nbytes(self%self_v), int64)
        self%fallback_backend = ggml_backend_for_cpu()
        if (.not. c_associated(self%fallback_backend)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper decoder CPU fallback backend allocation failed')
            call self%close()
            return
        end if
        ! Match whisper.cpp's preallocated decoder metadata arena.  Resetting
        ! this context for each graph avoids a host malloc/free cycle for
        ! every token while keeping graph tensors short-lived.
        self%decode_context = ggml_context_init(ggml_init_params(16_c_size_t * 1024_c_size_t * 1024_c_size_t, WHISPER_C_TRUE))
        if (.not. c_associated(self%decode_context)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper decoder graph metadata allocation failed')
            call self%close()
            return
        end if
    end subroutine whisper_decoder_init

    integer(int32) function whisper_pad_256(value)
        integer(int32), intent(in) :: value

        whisper_pad_256 = ((value + 255_int32) / 256_int32) * 256_int32
    end function whisper_pad_256

    subroutine whisper_decoder_prepare_cross(self, encoder_output, stat)
        class(whisper_decoder_t), intent(inout) :: self
        type(c_ptr), intent(in) :: encoder_output
        type(status_t), intent(out) :: stat
        type(c_ptr) :: context, graph, key, value, destination, encoder_input
        integer(int32) :: layer
        integer(c_int64_t) :: nstate, nctx, nctx_pad, nhead, nlayer, head_dim
        integer(c_size_t) :: offset
        integer(c_int) :: code
        character(len=128) :: name
        logical :: ok

        call stat%clear()
        if (.not. self%is_ready() .or. .not. c_associated(encoder_output)) then
            call stat%set(FORTAI_INVALID, 'Whisper decoder cross cache requires initialized state and encoder output')
            return
        end if
        call whisper_decoder_clear_cross_cache(self)
        nstate = int(self%n_state, c_int64_t)
        nctx = int(self%n_audio_ctx, c_int64_t)
        nctx_pad = int(self%n_audio_ctx_pad, c_int64_t)
        nhead = int(self%n_heads, c_int64_t)
        nlayer = int(self%n_layers, c_int64_t)
        head_dim = nstate / nhead
        context = ggml_context_init(ggml_init_params(16_c_size_t * 1024_c_size_t * 1024_c_size_t, WHISPER_C_TRUE))
        if (.not. c_associated(context)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper cross-attention graph metadata allocation failed')
            return
        end if
        graph = ggml_graph_new(context, int(WHISPER_GRAPH_NODES, c_size_t), WHISPER_C_FALSE)
        ! Reuse the encoder's backend allocation through a metadata-only view;
        ! the output is synchronized above and remains live for this graph.
        encoder_input = ggml_tensor_view(context, encoder_output)
        if (.not. c_associated(graph) .or. .not. c_associated(encoder_input)) then
            call ggml_context_free(context)
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper cross-attention graph allocation failed')
            return
        end if
        call ggml_tensor_set_input(encoder_input)
        do layer = 0_int32, self%n_layers - 1_int32
            call whisper_decoder_name(name, layer, 'cross_attn.key.weight')
            key = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), encoder_input)
            key = ggml_tensor_scale(context, key, real(head_dim, real32) ** (-0.25_real32))
            if (self%flash_attention) then
                offset = int(layer, c_size_t) * int(2_int64 * nstate * nctx_pad, c_size_t)
            else
                offset = int(layer, c_size_t) * int(2_int64 * nstate * nctx, c_size_t)
            end if
            destination = ggml_tensor_view_1d(context, self%cross_k, nstate * nctx, offset)
            call ggml_graph_expand(graph, ggml_tensor_cpy(context, key, destination))

            call whisper_decoder_name(name, layer, 'cross_attn.value.weight')
            value = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), encoder_input)
            call whisper_decoder_name(name, layer, 'cross_attn.value.bias')
            value = ggml_tensor_add(context, value, self%model%tensor_by_name(trim(name)))
            if (self%flash_attention) then
                destination = ggml_tensor_view_1d(context, self%cross_v, nstate * nctx, offset)
            else
                value = ggml_tensor_transpose(context, ggml_tensor_reshape_2d(context, value, nstate, nctx))
                destination = ggml_tensor_view_2d(context, self%cross_v, nctx, nstate, &
                    int(2_int64 * nctx, c_size_t), offset)
            end if
            call ggml_graph_expand(graph, ggml_tensor_cpy(context, value, destination))
        end do
        ! Reuse the decoder scheduler for cross-attention preparation.  The
        ! graph metadata is short-lived, but retaining the scheduler keeps its
        ! backend allocators and CUDA stream state hot across audio chunks.
        if (.not. c_associated(self%scheduler)) then
            self%scheduler = ggml_backend_sched_new(self%model%backend, int(WHISPER_GRAPH_NODES, c_size_t), WHISPER_C_FALSE, &
                WHISPER_C_TRUE, self%fallback_backend)
        end if
        if (.not. c_associated(self%scheduler)) then
            call ggml_context_free(context)
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper cross-attention scheduler allocation failed')
            return
        end if
        ! The encoder output is already resident on the model backend.  Pin
        ! this graph input there so the scheduler does not silently select its
        ! CPU fallback and insert a device-to-host-to-device staging copy.
        call ggml_backend_sched_reset(self%scheduler)
        call ggml_backend_sched_set_tensor_backend(self%scheduler, encoder_input, self%model%backend)
        ok = ggml_backend_sched_alloc_graph(self%scheduler, graph)
        if (.not. ok) then
            call ggml_backend_sched_reset(self%scheduler)
            call ggml_context_free(context)
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper cross-attention compute arena allocation failed')
            return
        end if
        code = ggml_backend_sched_graph_compute(self%scheduler, graph)
        call ggml_backend_sched_reset(self%scheduler)
        call ggml_context_free(context)
        if (code /= GGML_STATUS_SUCCESS) then
            call stat%set(FORTAI_UNSUPPORTED, 'Whisper cross-attention cache computation failed')
            return
        end if
        self%cross_ready = .true.
    end subroutine whisper_decoder_prepare_cross

    subroutine whisper_decoder_decode_tokens(self, tokens, position, logits, stat)
        class(whisper_decoder_t), intent(inout) :: self
        integer(int32), intent(in), target :: tokens(:)
        integer(int32), intent(in) :: position
        real(real32), allocatable, target, intent(inout) :: logits(:)
        type(status_t), intent(out) :: stat
        type(c_ptr) :: context, graph, token_tensor, position_tensor, mask_tensor, mask_f16, logits_tensor
        type(c_ptr) :: cur, inp_l, inp_ca, inp_ff, qcur, kcur, vcur, query, key, value, attended
        type(c_ptr) :: scores, probabilities, destination, cross_key, cross_value
        type(c_ptr) :: layer_k, layer_v, dweight, dbias
        integer(c_int64_t) :: nstate, nctx, nctx_pad, nhead, head_dim, n_kv, one, n_tokens, n_vocab
        integer(c_size_t) :: offset, bytes, stride1, stride2
        integer(int32) :: layer, i, j, token_count
        integer(int32), allocatable, target :: token_host(:), position_host(:)
        real(real32), allocatable, target :: mask_host(:)
        character(len=128) :: name
        integer(c_int) :: code
        logical :: ok

        call stat%clear()
        if (.not. self%is_ready() .or. .not. self%cross_ready) then
            call stat%set(FORTAI_INVALID, 'Whisper decoder requires a prepared cross-attention cache')
            return
        end if
        token_count = int(size(tokens), int32)
        if (token_count <= 0) then
            call stat%set(FORTAI_INVALID, 'Whisper decoder token batch is empty')
            return
        end if
        if (position < 0 .or. position >= self%n_text_ctx .or. token_count > self%n_text_ctx - position) then
            call stat%set(FORTAI_INVALID, 'Whisper decoder position is outside the context')
            return
        end if
        nstate = int(self%n_state, c_int64_t)
        nctx = int(self%n_text_ctx, c_int64_t)
        nctx_pad = int(self%n_text_ctx_pad, c_int64_t)
        nhead = int(self%n_heads, c_int64_t)
        head_dim = nstate / nhead
        one = 1_c_int64_t
        n_tokens = int(token_count, c_int64_t)
        n_vocab = int(self%model%file%hparams%n_vocab, c_int64_t)
        if (self%flash_attention .and. self%model%use_gpu) then
            n_kv = min(nctx, int(((position + token_count + 255_int32) / 256_int32) * 256_int32, c_int64_t))
        else
            n_kv = int(position + token_count, c_int64_t)
        end if
        context = self%decode_context
        call ggml_context_reset(context)
        if (.not. c_associated(context)) then
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper decoder graph metadata allocation failed')
            return
        end if
        graph = ggml_graph_new(context, int(WHISPER_GRAPH_NODES, c_size_t), WHISPER_C_FALSE)
        token_tensor = ggml_tensor_new_1d(context, GGML_TYPE_I32, n_tokens)
        position_tensor = ggml_tensor_new_1d(context, GGML_TYPE_I32, n_tokens)
        mask_tensor = ggml_tensor_new_3d(context, GGML_TYPE_F32, n_kv, n_tokens, one)
        call ggml_tensor_set_input(token_tensor)
        call ggml_tensor_set_input(position_tensor)
        call ggml_tensor_set_input(mask_tensor)
        mask_f16 = ggml_tensor_cast(context, mask_tensor, GGML_TYPE_F16)
        cur = ggml_tensor_add(context, &
            ggml_tensor_get_rows(context, self%model%tensor_by_name('decoder.token_embedding.weight'), token_tensor), &
            ggml_tensor_get_rows(context, self%model%tensor_by_name('decoder.positional_embedding'), position_tensor))
        inp_l = cur
        stride1 = int(2_int64 * nstate, c_size_t)
        stride2 = int(2_int64 * head_dim, c_size_t)
        do layer = 0_int32, self%n_layers - 1_int32
            call whisper_decoder_name(name, layer, 'attn_ln.weight')
            dweight = self%model%tensor_by_name(trim(name))
            call whisper_decoder_name(name, layer, 'attn_ln.bias')
            dbias = self%model%tensor_by_name(trim(name))
            cur = ggml_tensor_norm(context, inp_l, WHISPER_EPS)
            cur = ggml_tensor_add(context, ggml_tensor_mul(context, cur, dweight), dbias)

            call whisper_decoder_name(name, layer, 'attn.query.weight')
            qcur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            call whisper_decoder_name(name, layer, 'attn.query.bias')
            qcur = ggml_tensor_add(context, qcur, self%model%tensor_by_name(trim(name)))
            qcur = ggml_tensor_scale(context, qcur, real(head_dim, real32) ** (-0.25_real32))
            call whisper_decoder_name(name, layer, 'attn.key.weight')
            kcur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            kcur = ggml_tensor_scale(context, kcur, real(head_dim, real32) ** (-0.25_real32))
            call whisper_decoder_name(name, layer, 'attn.value.weight')
            vcur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            call whisper_decoder_name(name, layer, 'attn.value.bias')
            vcur = ggml_tensor_add(context, vcur, self%model%tensor_by_name(trim(name)))

            offset = int(layer, c_size_t) * int(2_int64 * nstate * nctx_pad, c_size_t)
            destination = ggml_tensor_view_1d(context, self%self_k, nstate * n_tokens, &
                offset + int(2_int64 * nstate * position, c_size_t))
            call ggml_graph_expand(graph, ggml_tensor_cpy(context, kcur, destination))
            if (self%flash_attention) then
                destination = ggml_tensor_view_1d(context, self%self_v, nstate * n_tokens, &
                    offset + int(2_int64 * nstate * position, c_size_t))
            else
                vcur = ggml_tensor_transpose(context, ggml_tensor_reshape_2d(context, vcur, nstate, n_tokens))
                destination = ggml_tensor_view_2d(context, self%self_v, n_tokens, nstate, &
                    int(2_int64 * nctx_pad, c_size_t), offset + int(2_int64 * position, c_size_t))
            end if
            call ggml_graph_expand(graph, ggml_tensor_cpy(context, vcur, destination))

            query = ggml_tensor_permute(context, ggml_tensor_reshape_3d(context, qcur, head_dim, nhead, n_tokens), &
                0_c_int, 2_c_int, 1_c_int, 3_c_int)
            if (self%flash_attention) then
                layer_k = ggml_tensor_view_3d(context, self%self_k, head_dim, n_kv, nhead, stride1, stride2, offset)
                layer_v = ggml_tensor_view_3d(context, self%self_v, head_dim, n_kv, nhead, stride1, stride2, offset)
                attended = ggml_tensor_flash_attn_ext(context, query, layer_k, layer_v, mask_f16, &
                    1.0_real32, 0.0_real32, 0.0_real32)
                cur = ggml_tensor_reshape_2d(context, attended, nstate, n_tokens)
            else
                layer_k = ggml_tensor_view_3d(context, self%self_k, head_dim, n_kv, nhead, stride1, stride2, offset)
                layer_v = ggml_tensor_view_3d(context, self%self_v, n_kv, head_dim, nhead, &
                    int(2_int64 * nctx_pad, c_size_t), int(2_int64 * nctx_pad * head_dim, c_size_t), offset)
                key = layer_k
                value = layer_v
                scores = ggml_tensor_mul_mat(context, key, query)
                probabilities = ggml_tensor_soft_max_ext(context, scores, mask_tensor, 1.0_real32, 0.0_real32)
                attended = ggml_tensor_mul_mat(context, value, probabilities)
                cur = ggml_tensor_cont_2d(context, ggml_tensor_permute(context, attended, 0_c_int, 2_c_int, 1_c_int, 3_c_int), &
                    nstate, n_tokens)
            end if
            call whisper_decoder_name(name, layer, 'attn.out.weight')
            cur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            call whisper_decoder_name(name, layer, 'attn.out.bias')
            cur = ggml_tensor_add(context, cur, self%model%tensor_by_name(trim(name)))
            inp_ca = ggml_tensor_add(context, cur, inp_l)

            call whisper_decoder_name(name, layer, 'cross_attn_ln.weight')
            dweight = self%model%tensor_by_name(trim(name))
            call whisper_decoder_name(name, layer, 'cross_attn_ln.bias')
            dbias = self%model%tensor_by_name(trim(name))
            cur = ggml_tensor_norm(context, inp_ca, WHISPER_EPS)
            cur = ggml_tensor_add(context, ggml_tensor_mul(context, cur, dweight), dbias)
            call whisper_decoder_name(name, layer, 'cross_attn.query.weight')
            qcur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            call whisper_decoder_name(name, layer, 'cross_attn.query.bias')
            qcur = ggml_tensor_add(context, qcur, self%model%tensor_by_name(trim(name)))
            query = ggml_tensor_permute(context, ggml_tensor_reshape_3d(context, qcur, head_dim, nhead, n_tokens), &
                0_c_int, 2_c_int, 1_c_int, 3_c_int)
            if (self%flash_attention) then
                offset = int(layer, c_size_t) * int(2_int64 * nstate * self%n_audio_ctx_pad, c_size_t)
            else
                offset = int(layer, c_size_t) * int(2_int64 * nstate * int(self%n_audio_ctx, c_int64_t), c_size_t)
            end if
            if (self%flash_attention) then
                cross_key = ggml_tensor_view_3d(context, self%cross_k, head_dim, int(self%n_audio_ctx_pad, c_int64_t), nhead, &
                    stride1, stride2, offset)
                cross_value = ggml_tensor_view_3d(context, self%cross_v, head_dim, int(self%n_audio_ctx_pad, c_int64_t), nhead, &
                    stride1, stride2, offset)
            else
                cross_key = ggml_tensor_view_3d(context, self%cross_k, head_dim, int(self%n_audio_ctx, c_int64_t), nhead, &
                    stride1, stride2, offset)
                cross_value = ggml_tensor_view_3d(context, self%cross_v, head_dim, int(self%n_audio_ctx, c_int64_t), nhead, &
                    stride1, stride2, offset)
            end if
            if (self%flash_attention) then
                attended = ggml_tensor_flash_attn_ext(context, query, cross_key, cross_value, c_null_ptr, &
                    real(head_dim, real32) ** (-0.25_real32), 0.0_real32, 0.0_real32)
                cur = ggml_tensor_reshape_2d(context, attended, nstate, n_tokens)
            else
                scores = ggml_tensor_mul_mat(context, cross_key, query)
                probabilities = ggml_tensor_soft_max_ext(context, scores, c_null_ptr, &
                    real(head_dim, real32) ** (-0.25_real32), 0.0_real32)
                value = ggml_tensor_view_3d(context, self%cross_v, int(self%n_audio_ctx, c_int64_t), head_dim, nhead, &
                    int(2_int64 * self%n_audio_ctx, c_size_t), &
                    int(2_int64 * self%n_audio_ctx * head_dim, c_size_t), &
                    int(layer, c_size_t) * int(2_int64 * nstate * int(self%n_audio_ctx, c_int64_t), c_size_t))
                attended = ggml_tensor_mul_mat(context, value, probabilities)
                cur = ggml_tensor_cont_2d(context, ggml_tensor_permute(context, attended, 0_c_int, 2_c_int, 1_c_int, 3_c_int), &
                    nstate, n_tokens)
            end if
            call whisper_decoder_name(name, layer, 'cross_attn.out.weight')
            cur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            call whisper_decoder_name(name, layer, 'cross_attn.out.bias')
            cur = ggml_tensor_add(context, cur, self%model%tensor_by_name(trim(name)))
            inp_ff = ggml_tensor_add(context, cur, inp_ca)

            call whisper_decoder_name(name, layer, 'mlp_ln.weight')
            dweight = self%model%tensor_by_name(trim(name))
            call whisper_decoder_name(name, layer, 'mlp_ln.bias')
            dbias = self%model%tensor_by_name(trim(name))
            cur = ggml_tensor_norm(context, inp_ff, WHISPER_EPS)
            cur = ggml_tensor_add(context, ggml_tensor_mul(context, cur, dweight), dbias)
            call whisper_decoder_name(name, layer, 'mlp.0.weight')
            cur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            call whisper_decoder_name(name, layer, 'mlp.0.bias')
            cur = ggml_tensor_add(context, cur, self%model%tensor_by_name(trim(name)))
            cur = ggml_tensor_gelu(context, cur)
            call whisper_decoder_name(name, layer, 'mlp.2.weight')
            cur = ggml_tensor_mul_mat(context, self%model%tensor_by_name(trim(name)), cur)
            call whisper_decoder_name(name, layer, 'mlp.2.bias')
            cur = ggml_tensor_add(context, cur, self%model%tensor_by_name(trim(name)))
            inp_l = ggml_tensor_add(context, cur, inp_ff)
        end do

        cur = ggml_tensor_norm(context, inp_l, WHISPER_EPS)
        cur = ggml_tensor_add(context, ggml_tensor_mul(context, cur, self%model%tensor_by_name('decoder.ln.weight')), &
            self%model%tensor_by_name('decoder.ln.bias'))
        cur = ggml_tensor_mul_mat(context, self%model%tensor_by_name('decoder.token_embedding.weight'), cur)
        logits_tensor = cur
        if (token_count > 1) then
            logits_tensor = ggml_tensor_view_2d(context, cur, n_vocab, one, &
                int(4_int64 * n_vocab, c_size_t), int(4_int64 * n_vocab * int(token_count - 1, int64), c_size_t))
        end if
        call ggml_tensor_set_output(logits_tensor)
        call ggml_graph_build(graph, logits_tensor)

        allocate(token_host(token_count), position_host(token_count))
        do j = 1, token_count
            token_host(j) = tokens(j)
            position_host(j) = position + j - 1
        end do
        allocate(mask_host(int(n_kv, int32) * token_count))
        do j = 0, token_count - 1
            do i = 0, int(n_kv, int32) - 1
                if (i <= position + j) then
                    mask_host(j * int(n_kv, int32) + i + 1) = 0.0_real32
                else
                    mask_host(j * int(n_kv, int32) + i + 1) = -huge(1.0_real32)
                end if
            end do
        end do
        if (.not. c_associated(self%scheduler)) then
            self%scheduler = ggml_backend_sched_new(self%model%backend, int(WHISPER_GRAPH_NODES, c_size_t), &
                WHISPER_C_FALSE, WHISPER_C_TRUE, &
                self%fallback_backend)
        end if
        if (.not. c_associated(self%scheduler)) then
            deallocate(token_host, position_host)
            deallocate(mask_host)
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper decoder scheduler allocation failed')
            return
        end if
        ok = ggml_backend_sched_alloc_graph(self%scheduler, graph)
        if (.not. ok) then
            deallocate(token_host, position_host)
            deallocate(mask_host)
            call ggml_backend_sched_reset(self%scheduler)
            call stat%set(FORTAI_OUT_OF_MEMORY, 'Whisper decoder compute arena allocation failed')
            return
        end if
        call ggml_backend_set_tensor(token_tensor, c_loc(token_host(1)), 0_c_size_t, &
            int(size(token_host), c_size_t) * int(storage_size(token_host(1)) / 8, c_size_t))
        call ggml_backend_set_tensor(position_tensor, c_loc(position_host(1)), 0_c_size_t, &
            int(size(position_host), c_size_t) * int(storage_size(position_host(1)) / 8, c_size_t))
        bytes = int(size(mask_host), c_size_t) * int(storage_size(mask_host(1)) / 8, c_size_t)
        call ggml_backend_set_tensor(mask_tensor, c_loc(mask_host(1)), 0_c_size_t, bytes)
        code = ggml_backend_sched_graph_compute(self%scheduler, graph)
        if (code == GGML_STATUS_SUCCESS) then
            if (.not. allocated(logits)) then
                allocate(logits(int(n_vocab, int32)))
            else if (size(logits) /= int(n_vocab, int32)) then
                deallocate(logits)
                allocate(logits(int(n_vocab, int32)))
            end if
            bytes = int(size(logits), c_size_t) * int(storage_size(logits(1)) / 8, c_size_t)
            call ggml_backend_get_tensor(logits_tensor, c_loc(logits(1)), 0_c_size_t, bytes)
        end if
        call ggml_backend_sched_reset(self%scheduler)
        deallocate(token_host, position_host)
        deallocate(mask_host)
        if (code /= GGML_STATUS_SUCCESS) then
            if (allocated(logits)) deallocate(logits)
            call stat%set(FORTAI_UNSUPPORTED, 'Whisper decoder graph execution failed')
        end if
    end subroutine whisper_decoder_decode_tokens

    subroutine whisper_decoder_decode(self, token, position, logits, stat)
        class(whisper_decoder_t), intent(inout) :: self
        integer(int32), intent(in) :: token, position
        real(real32), allocatable, target, intent(inout) :: logits(:)
        type(status_t), intent(out) :: stat
        integer(int32), target :: token_batch(1)

        token_batch(1) = token
        call whisper_decoder_decode_tokens(self, token_batch, position, logits, stat)
    end subroutine whisper_decoder_decode

    subroutine whisper_decoder_name(name, layer, suffix)
        character(len=*), intent(out) :: name
        integer(int32), intent(in) :: layer
        character(len=*), intent(in) :: suffix

        write(name, '("decoder.blocks.",i0,".",a)') layer, trim(suffix)
    end subroutine whisper_decoder_name

    subroutine whisper_decoder_clear_cross_cache(self)
        class(whisper_decoder_t), intent(inout) :: self

        if (c_associated(self%state_buffer)) call ggml_backend_buffer_clear(self%state_buffer, 0)
        self%cross_ready = .false.
    end subroutine whisper_decoder_clear_cross_cache

    subroutine whisper_decoder_close(self)
        class(whisper_decoder_t), intent(inout) :: self

        call ggml_backend_sched_free(self%scheduler)
        call ggml_context_free(self%decode_context)
        call ggml_backend_free_buffer(self%state_buffer)
        call ggml_context_free(self%state_context)
        call ggml_backend_free(self%fallback_backend)
        nullify(self%model)
        self%cross_k = c_null_ptr
        self%cross_v = c_null_ptr
        self%self_k = c_null_ptr
        self%self_v = c_null_ptr
        self%n_audio_ctx = 0_int32
        self%n_audio_ctx_pad = 0_int32
        self%n_text_ctx = 0_int32
        self%n_text_ctx_pad = 0_int32
        self%n_state = 0_int32
        self%n_heads = 0_int32
        self%n_layers = 0_int32
        self%cache_bytes = 0_int64
        self%compute_bytes = 0_int64
        self%cross_ready = .false.
    end subroutine whisper_decoder_close

    integer(int64) function whisper_decoder_memory_bytes(self)
        class(whisper_decoder_t), intent(in) :: self

        whisper_decoder_memory_bytes = self%cache_bytes + self%compute_bytes
    end function whisper_decoder_memory_bytes

    logical function whisper_decoder_is_ready(self)
        class(whisper_decoder_t), intent(in) :: self

        whisper_decoder_is_ready = associated(self%model) .and. c_associated(self%state_buffer)
    end function whisper_decoder_is_ready

end module fortai_whisper_decoder
