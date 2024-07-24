# Implementing the SDF sampling without the binaries provided by DeepSDF because they did not produce any output.
# Instead use the mesh_to_sdf package: https://github.com/marian42/mesh_to_sdf
import glob

import mesh_to_sdf

import os
import json
import logging
import trimesh
import numpy as np

def append_data_source_map(data_dir, name, source):
    data_source_map_filename = os.path.join(data_dir, ".datasources.json")

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


class NoMeshFileError(Exception):
    pass


class MultipleMeshFileError(Exception):
    pass


def preprocess(data_dir, source_dir, source_name, class_directories, number_of_points, extension):
    dest_dir = os.path.join(data_dir, source_name)

    print(
        "Preprocessing data from "
        + source_dir
        + " and placing the results in "
        + dest_dir
    )

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    number_of_points = int(number_of_points)
    ext = extension

    append_data_source_map(data_dir, source_name, source_dir)

    meshes_targets_and_specific_args = []

    for class_dir in class_directories:
        class_path = os.path.join(source_dir, class_dir)
        instance_ids = class_directories[class_dir]

        print(
            "Processing " + str(len(instance_ids)) + " instances of class " + class_dir
        )

        target_dir = os.path.join(dest_dir, class_dir)

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        for instance_id in instance_ids:

            processed_filepath = os.path.join(target_dir, instance_id + ".npz")

            try:

                mesh_filenames = list(glob.iglob(os.path.join(class_path , instance_id + ext)))
                print(os.path.join(class_path,instance_id + ext))
                if len(mesh_filenames) == 0:
                    raise NoMeshFileError()
                elif len(mesh_filenames) > 1:
                    raise MultipleMeshFileError()

                mesh_filename = mesh_filenames[0]

                specific_args = []

                meshes_targets_and_specific_args.append(
                    (
                        mesh_filename,
                        processed_filepath,
                        specific_args,
                    )
                )

            except NoMeshFileError:
                logging.warning("No mesh found for instance " + instance_id)
            except MultipleMeshFileError:
                logging.warning("Multiple meshes found for instance " + instance_id)
    count = 1
    for (
            mesh_filepath,
            target_filepath,
            specific_args,
    ) in meshes_targets_and_specific_args:
        mesh = trimesh.load(mesh_filepath, force="mesh")

        print(f"Started {count}: {target_filepath}")

        points, sdf, colours = mesh_to_sdf.sample_sdf_near_surface(mesh, number_of_points=number_of_points)
        reshaped = np.column_stack((points, sdf, colours))

        pos = reshaped[reshaped[:, 3] > 0]
        neg = reshaped[reshaped[:, 3] < 0]

        np.savez(target_filepath, pos=pos, neg=neg)

        print(f"Finished {count}: {target_filepath}")

        count += 1
