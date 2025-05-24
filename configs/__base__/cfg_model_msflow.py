from argparse import Namespace


class cfg_model_msflow(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model = Namespace()
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=False,
								 image_size=256)
