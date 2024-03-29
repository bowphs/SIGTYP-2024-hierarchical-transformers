from transformers import PretrainedConfig

class CharToWordDebertaEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        attention_head_size=64,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-7,
        relative_attention=True,
        max_relative_positions=-1,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type="p2c|c2p",
        position_biased_input=False,
        conv_kernel_size=0,
        conv_groups=1,
        conv_activation="gelu",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.norm_rel_ebd = norm_rel_ebd
        self.share_att_key = share_att_key
        self.position_biased_input = position_biased_input
        self.position_buckets = position_buckets

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.layer_norm_eps = layer_norm_eps

        self.conv_kernel_size = conv_kernel_size
        self.conv_groups = conv_groups
        self.conv_activation = conv_activation


class CharToWordDebertaConfig(PretrainedConfig):
    model_type = "ctw_deberta"

    def __init__(
        self,
        vocab_size=128100,
        type_vocab_size=0,
        initializer_range=0.02,
        pad_token_id=0,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        residual_word_embedding=False,
        intra_word_encoder={},
        inter_word_encoder={},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.intra_word_encoder = CharToWordDebertaEncoderConfig(**intra_word_encoder)
        self.inter_word_encoder = CharToWordDebertaEncoderConfig(**inter_word_encoder)
        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", self.inter_word_encoder.hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
        self.residual_word_embedding = residual_word_embedding
