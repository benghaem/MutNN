class Config():

    def __init__(self, store_fn, load_fn, batch_width, dataset_len):

        self.user_store_fn = store_fn
        self.user_load_fn = load_fn
        self.batch_width = batch_width
        self.dataset_len = dataset_len

        #pnopt
