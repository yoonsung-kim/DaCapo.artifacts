from emulator.model import Model
from emulator.classification_model import ClassificationModel


class ModelFactory:
    @classmethod
    def generate_model(cls, config, task_type, name, precision, batch_size, num_classes, device, freeze, weight_path=None) -> Model:
        task_type = task_type

        if task_type == "classification":
            model = ClassificationModel(config=config,
                                        name=name,
                                        precision=precision,
                                        batch_size=batch_size,
                                        num_classes=num_classes,
                                        device=device,
                                        freeze=freeze,
                                        weight_path=weight_path)
        elif task_type == "detection":
            raise ValueError(f"detection is not supported")
        else:
            raise ValueError(f"invalid task type: {task_type}")
        
        return model