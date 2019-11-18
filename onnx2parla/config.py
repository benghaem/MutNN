class Config():

    def __init__(self, store_fn, load_fn, batch_size, dataset_len):

        self.user_store_fn = store_fn
        self.user_load_fn = load_fn
        self.batch_size = batch_size
        self.dataset_len = dataset_len

        # pnopt
