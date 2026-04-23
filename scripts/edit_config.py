import argparse
import glob
import os

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

def edit_config(train_dir, eval_dir, batch_size, checkpoint, label_map, config_path, output_path):
    """
    Parses a `pipeline.config`, updates necessary fields (batch size, input paths, 
    fine-tune checkpoint, label map), and saves it.
    """
    print(f"Reading configuration from: {config_path}")
    
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with tf.io.gfile.GFile(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    
    # Obtain list of tfrecords
    train_files = glob.glob(os.path.join(train_dir, "*.tfrecord"))
    eval_files = glob.glob(os.path.join(eval_dir, "*.tfrecord"))
    
    if not train_files:
        print(f"Warning: No training tfrecords found in {train_dir}")
    if not eval_files:
        print(f"Warning: No eval tfrecords found in {eval_dir}")

    # 1. Update Train Config
    print("Updating training configuration...")
    pipeline_config.train_config.batch_size = batch_size
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    
    # Append random horizontal flip if not present as a base augmentation
    has_random_horizontal_flip = any(
        step.HasField('random_horizontal_flip') for step in pipeline_config.train_config.data_augmentation_options
    )
    if not has_random_horizontal_flip:
        new_step = pipeline_config.train_config.data_augmentation_options.add()
        new_step.random_horizontal_flip.SetInParent()

    # 2. Update Input Readers
    print("Updating input reader paths...")
    # Train input reader
    pipeline_config.train_input_reader.label_map_path = label_map
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = train_files

    # Eval input reader (can be multiple, we update the first one)
    if len(pipeline_config.eval_input_reader) > 0:
        pipeline_config.eval_input_reader[0].label_map_path = label_map
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = eval_files
    else:
        # If no eval reader exists, create one
        eval_reader = pipeline_config.eval_input_reader.add()
        eval_reader.label_map_path = label_map
        eval_reader.tf_record_input_reader.input_path[:] = eval_files

    # 3. Save modified config
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(output_path, "wb") as f:
        f.write(config_text.encode('utf-8'))
        
    print(f"Pipeline config updated successfully and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customize a pre-trained TF2 Object Detection pipeline.config")
    parser.add_argument("--train_dir", required=True, type=str, help="Directory with train .tfrecords")
    parser.add_argument("--eval_dir", required=True, type=str, help="Directory with eval .tfrecords")
    parser.add_argument("--batch_size", default=8, type=int, help="Training batch size")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to pre-trained model checkpoint (e.g. ckpt-0)")
    parser.add_argument("--label_map", required=True, type=str, help="Path to label_map.pbtxt")
    parser.add_argument("--config_path", required=True, type=str, help="Path to input pipeline.config")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save updated pipeline.config")
    
    args = parser.parse_args()
    
    edit_config(args.train_dir, args.eval_dir, args.batch_size, 
                args.checkpoint, args.label_map, args.config_path, args.output_path)
