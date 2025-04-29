import pandas as pd
import numpy as np
from scipy.stats import shapiro, zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

FIELD_TYPE_FLOAT = "float"
FIELD_TYPE_INT = "int"
FIELD_TYPE_STR = "str"
FIELD_ACTION_COPY = "copy"
FIELD_ACTION_IGNORE = "ignore"
FIELD_ACTION_ZSCORE = "zscore"
FIELD_ACTION_NORMALIZE = "normalize"
FIELD_ACTION_DUMMY = "dummy"
FIELD_ACTION_TARGET = "target"

META_TYPE_REGRESSION = "regression"
META_TYPE_BINARY_CLASSIFICATION = "binary-classification"
META_TYPE_CLASSIFICATION = "classification"

def isnumeric(datatype):
    return datatype in [FIELD_TYPE_FLOAT, FIELD_TYPE_INT]

def find_positive(s):
    s = set(s.str.upper().tolist())
    if len(s) != 2: return None
    if "+" in s and "-" in s: return "+"
    if "0" in s and "1" in s: return "1"
    if "t" in s and "f" in s: return "t"
    if "y" in s and "n" in s: return "y"
    if "true" in s and "false" in s: return "true"
    if "yes" in s and "no" in s: return "yes"
    if "p" in s and "n" in s: return "p"
    if "positive" in s and "negative" in s: return "positive"
    s = list(s)
    s.sort()
    return s[0]

def analyze(df, target, is_regression=True):
    metadata = {
        "fields": {},
        "target": target,
        "source": "uploaded_file",
        "early_stop": True
    }
    fields = metadata["fields"]
    for field_name, csv_type in zip(df.columns, df.dtypes):
        if "float" in csv_type.name:
            dtype = FIELD_TYPE_FLOAT
            action = FIELD_ACTION_COPY
        elif "int" in csv_type.name:
            dtype = FIELD_TYPE_INT
            action = FIELD_ACTION_COPY
        else:
            dtype = FIELD_TYPE_STR
            action = FIELD_ACTION_IGNORE
        missing_count = sum(df[field_name].isnull())
        col = df[field_name]
        unique_count = len(pd.unique(col))
        if isnumeric(dtype):
            stat, p = shapiro(col)
            action = FIELD_ACTION_ZSCORE if p>0.05 else FIELD_ACTION_NORMALIZE
            fields[field_name] = {
                "type": dtype,
                "median": col.median(),
                "mean": col.mean(),
                "sd": col.std(),
                "max": col.max(),
                "min": col.min(),
                "shapiro_stat": stat,
                "shapiro_p": p,
                "action": action,
                "missing": missing_count,
                "unique": unique_count
            }
        else:
            fields[field_name] = {
                "type": dtype,
                "mode": col.mode()[0],
                "action": action,
                "missing": missing_count,
                "unique": unique_count
            }
        field = fields[field_name]
        if (field["type"] == FIELD_TYPE_STR) and (field["unique"] < 1000) and (field["unique"]/len(df) < 0.75):
            field["action"] = FIELD_ACTION_DUMMY
        if field_name == target:
            field["action"] = FIELD_ACTION_TARGET
    is_binary = (metadata["fields"][target]["unique"] == 2) and not is_regression
    if is_regression:
        metadata["type"] = META_TYPE_REGRESSION
    else:
        if metadata["fields"][target]["unique"] == 2:
            metadata["type"] = META_TYPE_BINARY_CLASSIFICATION
            metadata["positive_token"] = find_positive(df[target])
        else:
            metadata["type"] = META_TYPE_CLASSIFICATION
    return metadata

def preprocess_data(df, metadata):
    target = metadata["target"]
    x_fields = []
    df_processed = df.copy()
    for field_name in metadata["fields"]:
        field = metadata["fields"][field_name]
        if field["missing"] > 0:
            if isnumeric(field["type"]):
                df_processed[field_name] = df_processed[field_name].fillna(df_processed[field_name].median())
            else:
                df_processed[field_name] = df_processed[field_name].fillna(df_processed[field_name].mode()[0])
        if field["type"] == FIELD_TYPE_STR and field_name != target:
            try:
                df_processed[field_name] = pd.to_numeric(df_processed[field_name], errors='ignore')
                if pd.api.types.is_numeric_dtype(df_processed[field_name]):
                    field["type"] = FIELD_TYPE_FLOAT
            except:
                pass
        if field["action"] == FIELD_ACTION_ZSCORE and isnumeric(field["type"]):
            df_processed[field_name] = zscore(df_processed[field_name])
            x_fields.append(field_name)
        elif field["action"] == FIELD_ACTION_NORMALIZE and isnumeric(field["type"]):
            df_processed[field_name] = MinMaxScaler().fit_transform(df_processed[[field_name]])
            x_fields.append(field_name)
        elif field["action"] == FIELD_ACTION_DUMMY and field["type"] == FIELD_TYPE_STR:
            dummies = pd.get_dummies(df_processed[field_name], prefix=field_name, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            x_fields += dummies.columns.tolist()
        elif field["action"] == FIELD_ACTION_COPY and isnumeric(field["type"]):
            x_fields.append(field_name)
    for field in x_fields:
        if not pd.api.types.is_numeric_dtype(df_processed[field]):
            df_processed[field] = pd.to_numeric(df_processed[field], errors='coerce')
            df_processed[field] = df_processed[field].fillna(df_processed[field].median())
    x = df_processed[x_fields].values.astype(np.float32)
    if metadata["type"] == META_TYPE_CLASSIFICATION:
        dummies = pd.get_dummies(df_processed[target])
        y = dummies.values.astype(np.float32)
    elif metadata["type"] == META_TYPE_BINARY_CLASSIFICATION:
        pos = metadata["positive_token"]
        df_processed[target] = (df_processed[target] == pos).astype(int)
        y = df_processed[target].values.astype(np.float32)
    else:
        if not pd.api.types.is_numeric_dtype(df_processed[target]):
            df_processed[target] = pd.to_numeric(df_processed[target], errors='coerce')
            df_processed[target] = df_processed[target].fillna(df_processed[target].median())
        y = df_processed[target].values.astype(np.float32)
    return x, y, x_fields

def adjust_model_for_constraints(metadata, training_time_minutes, target_accuracy, max_model_size_mb, device):
    metadata["input_dim"] = len(metadata["fields"]) - 1
    max_params = int(max_model_size_mb * 250000)
    if max_params < 100000:
        metadata["hidden_units"] = [16, 8]
        metadata["dropout_rate"] = 0.2
        metadata["model_size"] = "small"
    elif max_params < 500000:
        metadata["hidden_units"] = [32, 16]
        metadata["dropout_rate"] = 0.3
        metadata["model_size"] = "medium"
    else:
        metadata["hidden_units"] = [64, 32]
        metadata["dropout_rate"] = 0.4
        metadata["model_size"] = "large"
    max_epochs = int(training_time_minutes * 60)
    metadata["epochs"] = min(max_epochs, 1000)
    if target_accuracy > 90:
        metadata["patience"] = 15
        metadata["min_delta"] = 1e-5
        metadata["learning_rate"] = 0.0001
        metadata["accuracy_level"] = "high"
    elif target_accuracy > 80:
        metadata["patience"] = 10
        metadata["min_delta"] = 1e-4
        metadata["learning_rate"] = 0.001
        metadata["accuracy_level"] = "medium"
    else:
        metadata["patience"] = 5
        metadata["min_delta"] = 1e-3
        metadata["learning_rate"] = 0.01
        metadata["accuracy_level"] = "low"
    if device == "Jetson Nano":
        metadata["batch_size"] = 32
        metadata["device_type"] = "gpu"
    elif device == "Google Coral Dev Board":
        metadata["batch_size"] = 64
        metadata["device_type"] = "tpu"
    else:
        metadata["batch_size"] = 16
        metadata["device_type"] = "cpu"
    if metadata["type"] == META_TYPE_REGRESSION:
        metadata["output_units"] = 1
    elif metadata["type"] == META_TYPE_BINARY_CLASSIFICATION:
        metadata["output_units"] = 1
    else:
        metadata["output_units"] = metadata["fields"][metadata["target"]]["unique"]
    return metadata

def train_model(x, y, metadata):
    if metadata["type"] == META_TYPE_REGRESSION:
        loss = "mean_squared_error"
    elif metadata["type"] == META_TYPE_BINARY_CLASSIFICATION:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"
    model = Sequential()
    model.add(Dense(metadata["hidden_units"][0], input_dim=x.shape[1], activation="relu"))
    model.add(Dropout(metadata["dropout_rate"]))
    model.add(Dense(metadata["hidden_units"][1], activation="relu"))
    model.add(Dropout(metadata["dropout_rate"]))
    if metadata["type"] == META_TYPE_REGRESSION:
        model.add(Dense(1))
    elif metadata["type"] == META_TYPE_BINARY_CLASSIFICATION:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(y.shape[1], activation="softmax"))
    optimizer = Adam(learning_rate=metadata["learning_rate"])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    if metadata["early_stop"]:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        monitor = EarlyStopping(
            monitor="val_loss", 
            min_delta=metadata["min_delta"], 
            patience=metadata["patience"], 
            verbose=1, 
            mode="auto", 
            restore_best_weights=True
        )
        history = model.fit(
            x_train, y_train, 
            validation_data=(x_test, y_test),
            callbacks=[monitor], 
            verbose=2, 
            epochs=metadata["epochs"],
            batch_size=metadata["batch_size"]
        )
    else:
        history = model.fit(
            x, y, 
            verbose=2, 
            epochs=metadata["epochs"],
            batch_size=metadata["batch_size"]
        )
    return model, history

def generate_training_script(df, metadata, x_fields):
    script = []
    script.append("import pandas as pd")
    script.append("import numpy as np")
    script.append("from scipy.stats import zscore")
    script.append("from sklearn.preprocessing import MinMaxScaler")
    script.append("from sklearn.model_selection import train_test_split")
    script.append("from tensorflow.keras.models import Sequential")
    script.append("from tensorflow.keras.layers import Dense, Dropout")
    script.append("from tensorflow.keras.optimizers import Adam")
    script.append("from tensorflow.keras.callbacks import EarlyStopping")
    script.append("")
    script.append("# Load and prepare data")
    script.append(f"df = pd.read_csv('your_data.csv', na_values=['NA', '?'])")
    script.append("")
    script.append("# Preprocessing steps")
    for field_name in metadata["fields"]:
        field = metadata["fields"][field_name]
        if field["missing"] > 0:
            if isnumeric(field["type"]):
                script.append(f"df['{field_name}'] = df['{field_name}'].fillna(df['{field_name}'].median())")
            else:
                script.append(f"df['{field_name}'] = df['{field_name}'].fillna(df['{field_name}'].mode()[0])")
        if field["action"] == FIELD_ACTION_ZSCORE and isnumeric(field["type"]):
            script.append(f"df['{field_name}'] = zscore(df['{field_name}'])")
        elif field["action"] == FIELD_ACTION_NORMALIZE and isnumeric(field["type"]):
            script.append(f"df['{field_name}'] = MinMaxScaler().fit_transform(df[['{field_name}']])")
        elif field["action"] == FIELD_ACTION_DUMMY and field["type"] == FIELD_TYPE_STR:
            script.append(f"dummies = pd.get_dummies(df['{field_name}'], prefix='{field_name}', drop_first=True)")
            script.append("df = pd.concat([df, dummies], axis=1)")
    script.append("")
    script.append("# Select features")
    script.append(f"x_fields = {x_fields}")
    script.append("x = df[x_fields].values.astype(np.float32)")
    script.append("")
    target = metadata["target"]
    if metadata["type"] == META_TYPE_CLASSIFICATION:
        script.append("# Prepare target for classification")
        script.append(f"dummies = pd.get_dummies(df['{target}'])")
        script.append("y = dummies.values.astype(np.float32)")
    elif metadata["type"] == META_TYPE_BINARY_CLASSIFICATION:
        script.append("# Prepare target for binary classification")
        pos = metadata["positive_token"]
        script.append(f"df['{target}'] = (df['{target}'] == '{pos}').astype(int)")
        script.append(f"y = df['{target}'].values.astype(np.float32)")
    else:
        script.append("# Prepare target for regression")
        script.append(f"y = df['{target}'].values.astype(np.float32)")
    script.append("")
    script.append("# Create model")
    script.append("model = Sequential()")
    script.append(f"model.add(Dense({metadata['hidden_units'][0]}, input_dim=x.shape[1], activation='relu'))")
    script.append(f"model.add(Dropout({metadata['dropout_rate']}))")
    script.append(f"model.add(Dense({metadata['hidden_units'][1]}, activation='relu'))")
    script.append(f"model.add(Dropout({metadata['dropout_rate']}))")
    if metadata["type"] == META_TYPE_REGRESSION:
        script.append("model.add(Dense(1))")
        loss = "mean_squared_error"
    elif metadata["type"] == META_TYPE_BINARY_CLASSIFICATION:
        script.append("model.add(Dense(1, activation='sigmoid'))")
        loss = "binary_crossentropy"
    else:
        script.append(f"model.add(Dense(y.shape[1], activation='softmax'))")
        loss = "categorical_crossentropy"
    script.append("")
    script.append("# Compile and train model")
    script.append(f"optimizer = Adam(learning_rate={metadata['learning_rate']})")
    script.append(f"model.compile(loss='{loss}', optimizer=optimizer, metrics=['accuracy'])")
    if metadata["early_stop"]:
        script.append("x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)")
        script.append("monitor = EarlyStopping(monitor='val_loss', min_delta=metadata['min_delta'], patience=metadata['patience'], verbose=1, mode='auto', restore_best_weights=True)")
        script.append("history = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=2, epochs=metadata['epochs'], batch_size=metadata['batch_size'])")
    else:
        script.append("history = model.fit(x, y, verbose=2, epochs=metadata['epochs'], batch_size=metadata['batch_size'])")
    script.append("")
    script.append("# Save model")
    script.append("model.save('automl_model.h5')")
    return "\n".join(script) 