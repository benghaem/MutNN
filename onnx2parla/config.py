class Config:
    def __init__(self, store_fn, load_fn, user_width, dataset_len):

        self.user_store_fn = store_fn
        self.user_load_fn = load_fn
        self.dataset_len = dataset_len

        self.user_width = user_width
        self.computed_batch_size = user_width

        self.debug_passes = False
        self.use_simple_model_para = False
        self.use_data_para = True
        self.model_id = None
