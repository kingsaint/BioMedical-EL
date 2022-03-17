
class MemoryEfficientSaver:
    def __init__(self,atomic_components_directory):
        self._subcomponent_gatherer = SubcomponentGatherer()
        self._atomic_component_saver = AtomicSubcomponentSaver(atomic_components_directory)
    def base_save(self,filepath):#base case for saving atomic components, only atomic components are saved to disk.
        raise NotImplementedError
    @property
    def subconfig(self,savable_component):#dictionary
        #takes savable component configs and parameters and produces a config dictionary
        class_name = self.__class__.__name__
        subcomponents,parameters = self.subcomponent_gatherer.get_args()
        config = {class_name:{"parameters":parameters}}
        if subcomponents:
            subcomponent_configs = [subcomponent.subconfig for subcomponent in self.savable_subcomponents]
            config[class_name] = config[class_name].union({"subcomponents":subcomponent_configs})
        return config
    def save(self,config_filepath,atomic_components_directory):
        config = self.subconfig(self).union({"subcomponents_directory":subcomponents_directory})
        #save config as yaml and save out the components
        self.save_subcomponents(self,directory_path)
    def save_subcomponents(self,directory_path):
        if not self.savable_components:#base case
            self.base_save()
        else:
            for savable_component in self.savable_components:
                savable_component.save_subcomponents(directory_path)
    
        
    def load(self,filepath):
        pass
        #load config
        #load subcomponents according to the config.
class SubcomponentGatherer:
    def get_args(self,component):
        class_name = component.__class__.__name__
        if class_name == "DualEmbedderEntityLinker":
            subcomponents,parameters = self.visit_dual_embedder(component)
        elif class_name == "BertConceptEmbedder":
            subcomponents,parameters = self.visit_concept_embedder(component)
        elif:
            pass
        elif:
            pass
        else:
            return [],[]
    def visit_dual_embedder(self,dual_embedder):
        subcomponents = {dual_embedder.document_embedder,dual_embedder.concept_embedder}
        parameters = {}
        return subcomponents,parameters
    def visit_concept_embedder(self,concept_embedder):
        subcomponents = {"lkb","bert_model","tokenizer"}
        parameters = {"max_seq_len":concept_embedder.max_seq_len,"lower_case":concept_embedder.lower_case}
        return subcomponents,parameters
    def visit_document_embedder(self,document_embedder):
        subcomponents = {"span_detector","tokenizer"}
        parameters = {"max_seq_len":document_embedder.max_seq_len,"lower_case":document_embedder.lower_case}
        return subcomponents,parameters
    def visit_span_detector(self,span_detector):
        subcomponents = {"bert_model","linear_classifier"}
        parameters = {"max_mention_length"}
        return subcomponents,parameters

class AtomicSubcomponentSaver:
    def __init__(self,atomic_components_directory):
        self._atomic_components_directory = atomic_components_directory
    def set_save_directory(self,save_directory):
   
    def save(self,component):
        class_name = self.__class__.__name__
        #get a unique filename for each object
        #if file exists, skip the write. 
        if class_name == "Bert_Model":
            os.path.join(self._atomic_components_directory)
        elif class_name == "Tokenizer":
            filepath = 
        elif class_name == "LKB"

class SavableComponent: 
    def memory_efficient_save(self,config_filepath,atomic_components_directory):
        memory_efficient_serializer = MemoryEfficientSaver(atomic_components_directory)
        MemoryEfficientSaver.save(self,config_filepath)
    def save(self,directory):
        raise NotImplementedError


    
