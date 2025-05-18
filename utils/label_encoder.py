import json

class LabelEncoder:
    def __init__(self, schema_path):
        """
        Initialize the LabelEncoder with a schema file.

        Args:
            schema_path (str): Path to the JSON schema file.
        """
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        self.feature_classes = self._flatten_schema()

    def _flatten_schema(self):
        """
        Flatten the schema to create a list of all feature classes.

        Returns:
            Dict: A dict of all feature classes.
        """
        flat = {}
        offset = 0
        for category, values in self.schema.items():
            flat[category] = {
                "values": values,
                "offset": offset,
                "size": len(values)
            }
            offset += len(values)  
        self.total_dim = offset
        return flat

    def encode(self, label_dict):
        """
        Encode a label dictionary into a one-hot encoded vector.

        Args:
            label_dict (dict): A dictionary of labels to encode.

        Returns:
            List: A one-hot encoded vector.
        """
        one_hot = [0] * self.total_dim
        for feature, value in label_dict.items():
            if feature in self.feature_classes:
                idx = self.feature_classes[feature]["values"].index(value)
                one_hot[self.feature_classes[feature]["offset"] + idx] = 1
            else:
                continue
        return one_hot
    
    def decode(self, one_hot_vector):
        """
        Decode a one-hot encoded vector back into a label dictionary.

        Args:
            one_hot_vector (List): A one-hot encoded vector.

        Returns:
            Dict: A dictionary of decoded labels.
        """
        labels = {}
        for feature, info in self.feature_classes.items():
            start = info["offset"]
            end = start + info["size"]
            values = info["values"]
            sub_vector = one_hot_vector[start:end]
            max_idx = sub_vector.index(max(sub_vector))
            labels[feature] = values[max_idx]
        return labels