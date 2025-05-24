from argparse import Namespace


class cfg_model_inpformer(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model = Namespace()

		self.model.name = 'inpformer'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True,
								 encoder_arch='base', INP_num=6)