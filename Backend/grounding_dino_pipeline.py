import torch
import requests
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

OBB_IMAGE_PATH = "/home/Ubuntu/SUPARCO/Backend/public/image_with_obb.jpg"

def run_grounding_dino(grounding_dino_model, grounding_dino_processor, image, text_prompt):
    
    print(text_prompt)
    print(image)
    
    inputs = grounding_dino_processor(images=image, text=text_prompt, return_tensors="pt").to(grounding_dino_model.device)

    with torch.no_grad():
        outputs = grounding_dino_model(**inputs)

    # Post-process to get boxes
    target_sizes = torch.tensor([image.size[::-1]]) # [height, width]
    results = grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold = 0.4,
        target_sizes=target_sizes
    )

    # Get ALL detected boxes, not just the first one
    all_aabb_boxes = results[0]['boxes']

    if len(all_aabb_boxes) == 0:
        print("Grounding DINO found 0 boxes for the prompt.")
        boxes_found = False
        return all_aabb_boxes
    else:
        boxes_found = True
        print(f"Grounding DINO found {len(all_aabb_boxes)} boxes.")

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for box in all_aabb_boxes:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue AABB

        # plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        # plt.title(f"Step 2: Grounding DINO Output ({len(all_aabb_boxes)} AABBs)")
        # plt.axis('off')
        # plt.show()
        output_path = "/home/Ubuntu/SUPARCO/Backend/public/image_with_boxes.jpg"
        cv2.imwrite(output_path, image_cv)
        print(f"Saved image with boxes to {output_path}")
        return all_aabb_boxes


def run_sam_on_boxes(sam_model, sam_processor, image, gd_boxes):

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    all_binary_masks_cv = []
    green_overlay = np.zeros_like(image_cv, dtype=np.uint8)
    
    # print(f"Running SAM for {len(all_aabb_boxes)} boxes (one by one)...")

    # Loop through each box found by Grounding DINO
    for aabb_box in gd_boxes:
        
        # Process inputs for SAM 
        input_boxes = [[aabb_box.tolist()]]
        
        sam_inputs = sam_processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(sam_model.device)

        # Run model
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)

        # Post-process to get the mask
        # reshaped_hw = sam_inputs["pixel_values"].shape[2:]
        # reshaped_input_sizes = torch.tensor([reshaped_hw]).to(sam_model.device)
        reshaped_input_sizes = sam_inputs["reshaped_input_sizes"].to(sam_model.device)

        
        mask = sam_processor.post_process_masks(
            sam_outputs.pred_masks,
            sam_inputs["original_sizes"].to(sam_model.device),
            reshaped_input_sizes, 
            binarize=True
        )[0]

        # Extract the binary mask (True/False)
        binary_mask_bool = mask[0][0].cpu().numpy()

        # Convert to OpenCV format (0 or 255)
        binary_mask_cv = (binary_mask_bool * 255).astype(np.uint8)
        
        # Add the mask to our list and the overlay
        all_binary_masks_cv.append(binary_mask_cv)
        green_overlay[binary_mask_bool] = [0, 255, 0] # Green
    
    print(f"SAM generated {len(all_binary_masks_cv)} masks.")
    mask_found = True
    
    # Visualize ALL Masks
    image_with_mask = cv2.addWeighted(image_cv, 0.7, green_overlay, 0.3, 0)
    # plt.imshow(cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB))
    # plt.title(f"Step 3: SAM Output ({len(all_binary_masks_cv)} Masks)")
    # plt.axis('off')
    # plt.show()
    output_path = "/home/Ubuntu/SUPARCO/Backend/public/image_with_mask.jpg"
    cv2.imwrite(output_path, image_with_mask)
    print(f"Saved image with boxes to {output_path}")
    return all_binary_masks_cv

def generate_obb_from_mask(image, all_binary_masks_cv):
    all_obb_points = []
    image_final = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Loop through every binary mask we found
    for binary_mask_cv in all_binary_masks_cv:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the oriented bounding box
            oriented_box = cv2.minAreaRect(largest_contour)

            # Get the 4 corner points of the OBB
            box_points = cv2.boxPoints(oriented_box)
            box_points = np.intp(box_points) # Convert to integer points
            
            # Add the points to our list
            all_obb_points.append(box_points)

    print(f"Calculated {len(all_obb_points)} oriented boxes.")

    # Draw ALL contours found
    cv2.drawContours(image_final, all_obb_points, -1, (0, 0, 255), 2) # -1 means draw all
    # output_path = "/home/Ubuntu/SUPARCO/Backend/public/image_with_obb.jpg"
    cv2.imwrite(OBB_IMAGE_PATH, image_final)
    return all_obb_points
    

def generate_obb_pipeline(grounding_dino_model, grounding_dino_processor, sam_model, sam_processor, image_path, text_prompt):
    image = Image.open(image_path).convert("RGB")
    boxes = run_grounding_dino(grounding_dino_model, grounding_dino_processor, image, text_prompt)
    # print(boxes)

    if len(boxes)==0:
        return [], None
    
    masks = run_sam_on_boxes(sam_model, sam_processor, image, boxes)
    # print(masks)

    if len(masks)==0:
        return [], None

    obb_coordinates = generate_obb_from_mask(image, masks)
    print(obb_coordinates)
    obb_coordinates_list = [box.tolist() for box in obb_coordinates]
    return obb_coordinates_list, OBB_IMAGE_PATH



# grounding_dino_model, grounding_dino_processor, sam_model, sam_processor = load_grounding_models()
# image_url_satellite = "https://media.istockphoto.com/id/1285519123/photo/aerial-drone-photo-of-air-planes-as-seen-from-above-docked-in-airport-space.jpg?s=612x612&w=0&k=20&c=rfwg7mjvFtY9b9Hvv0yiEm1ro_fJWxe0QNywnflyM4w="
# image_url = "https://www.shutterstock.com/image-photo/airplane-flying-sky-travel-background-600nw-2478264897.jpg"
# imgresponse = requests.get(image_url_satellite)
# image_path = "/home/Ubuntu/SUPARCO/Backend/public/image2.jpg"

# with open(image_path, "wb") as f:
#     f.write(imgresponse.content)

# image = Image.open(image_path).convert("RGB")
# boxes = run_grounding_dino(grounding_dino_model, grounding_dino_processor, image, "an airplane in the moddle of the runway")
# masks = run_sam_on_boxes(sam_model, sam_processor, image, boxes)
# generate_obb_from_mask(image, masks)


