
import yaml
import os
from transformers import BertModel,BertTokenizer
import torch.nn as nn
import torch

class SubcomponentGatherer:#Visitor, may just actually just add arg gathering as part of SavableComponent interface.
    def visit_dual_embedder(self,dual_embedder):
        subcomponents = [dual_embedder.concept_embedder,dual_embedder.document_embedder]
        parameters = {"knowledge_data_path":dual_embedder.knowledge_data_path}
        return subcomponents,parameters
    def visit_concept_embedder(self,concept_embedder):
        subcomponents = [concept_embedder.bert_model,concept_embedder.tokenizer]
        parameters = {"max_ent_len":concept_embedder.max_ent_len,"lower_case":concept_embedder.lower_case}
        return subcomponents,parameters
    def visit_document_embedder(self,document_embedder):
        subcomponents = [document_embedder.span_detector,document_embedder.tokenizer]
        parameters = {"max_seq_len":document_embedder.max_seq_len,"lower_case":document_embedder.lower_case}
        return subcomponents,parameters
    def visit_span_detector(self,span_detector):
        subcomponents = [span_detector.bert_model,span_detector.linear_classifier]
        parameters = {"max_mention_len":span_detector.max_mention_len}
        return subcomponents,parameters
        

class AtomicSubcomponentSaver:
    def __init__(self,file_path):
        self._file_path = file_path
    def visit_bert_module(self,bert_model):
        torch.save(bert_model,self._file_path)
    def visit_linear_module(self,linear_model):
        torch.save(linear_model,self._file_path)
    def visit_tokenizer(self,tokenizer):
        tokenizer.save_pretrained(self._file_path)

class AtomicBaseNameGetter:
    def visit_bert_module(self,bert_model):
        return "bert_model.pt"
    def visit_linear_module(self,linear_model):
        return "linear_model.pt"
    def visit_tokenizer(self,tokenizer):
        return "tokenizer/"

class AtomicSubcomponentLoader:
    def __init__(self,file_path):
        self._file_path = file_path
    def visit_bert_module(self,bert_model):
        return torch.load(self._file_path)
    def visit_linear_module(self,linear_model):
        return torch.load(self._file_path)
    def visit_tokenizer(self,tokenizer):
        return tokenizer.from_pretrained(self._file_path)

class SavableComponent: 
    def save(self,directory_path):#dictionary
        #takes savable component configs and parameters and produces a config dictionary
        assert not os.path.isdir(directory_path)
        self._save(directory_path,True)
    def _save(self,directory_path,top_level):
        class_name = self.__class__.__name__
        if top_level:
            component_path = directory_path
        else:
            component_path = os.path.join(directory_path,class_name)
        config = {"component_class":class_name}

        if self.atomic():
            basename = self.accept(AtomicBaseNameGetter())
            file_path = os.path.join(directory_path,basename)
            self.accept(AtomicSubcomponentSaver(file_path))
        else:
            os.mkdir(component_path)
            subcomponents,parameters = self.accept(SubcomponentGatherer())
            subcomponent_configs = [subcomponent._save(component_path,False) for subcomponent in subcomponents]
            config["subcomponents"] = subcomponent_configs
            config["parameters"] = parameters 
        if top_level:
            config_path = os.path.join(directory_path,"config.yaml")
            with open(config_path, 'w') as outfile:
                yaml.dump(config,outfile,indent = 2)
        else:
            return config
    @classmethod
    def load(cls,directory_path):
        import el_toolkit.entity_linkers.dual_embedder.concept_embedder as concept_embedder 
        import el_toolkit.entity_linkers.dual_embedder.document_embedder as document_embedder 
        import el_toolkit.entity_linkers.dual_embedder.entity_linker as entity_linker
        import el_toolkit.entity_linkers.dual_embedder.model as model 
        LOADABLE_CLASSES = {
                    concept_embedder.BertConceptEmbedder,
                    document_embedder.DocumentEmbedder,
                    entity_linker.DualEmbedderEntityLinker,
                    model.BertMentionDetectorModel,
                    model.DualEmbedderModel,
                    WrappedBertModel,
                    WrappedLinearModule,
                    WrappedBertTokenizer
                    }

        CLASS_DICT = {cls.__name__:cls for cls in LOADABLE_CLASSES}
        def _load(config,directory_path,top_level):
            class_name = config["component_class"]
            print(class_name)
            component_class =  CLASS_DICT[config["component_class"]]
            if component_class.atomic():
                component_base_name = component_class.class_accept(AtomicBaseNameGetter())
                file_path = os.path.join(directory_path,component_base_name)
                return component_class.class_accept(AtomicSubcomponentLoader(file_path))
            else:
                if top_level:
                    subcomponent_path = directory_path
                else:
                    subcomponent_path = os.path.join(directory_path,class_name)
                subcomponents = []
                for subcomponent_config in config["subcomponents"]:
                    subcomponents.append(_load(subcomponent_config,subcomponent_path,False))
                parameters = config["parameters"]
                print(len(subcomponents))
                return component_class(*subcomponents,**parameters)    
        config_path = os.path.join(directory_path,"config.yaml")
        config = SavableComponent.load_config(config_path)
        return _load(config,directory_path,True) 
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    def accept(self,visitor):
        raise NotImplementedError
    @classmethod
    def class_accept(cls,visitor):
        raise NotImplementedError

class SavableAtomicComponent(SavableComponent): 
    @classmethod
    def atomic(cls):
        return True

class SavableCompositeComponent(SavableComponent):
    @classmethod
    def atomic(cls):
        return False

class WrappedBertModel(BertModel,SavableAtomicComponent):
    def accept(self,visitor):
        return visitor.visit_bert_module(self)
    @classmethod
    def class_accept(cls,visitor):
        return visitor.visit_bert_module(cls)
class WrappedLinearModule(nn.Linear,SavableAtomicComponent):
    def accept(self,visitor):
        return visitor.visit_linear_module(self)
    @classmethod
    def class_accept(cls,visitor):
        return visitor.visit_linear_module(cls)
class WrappedBertTokenizer(BertTokenizer,SavableAtomicComponent):
    def accept(self,visitor):
        return visitor.visit_tokenizer(self)
    @classmethod
    def class_accept(cls,visitor):
        return visitor.visit_tokenizer(cls)