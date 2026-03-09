import os
import numpy as np
import argparse
import json

def parser():
    ap = argparse.ArgumentParser(description='DeSite for BSU')

    ap.add_argument('--ASMfolder', type=str, default='./output/ASM/', help='Read the ASM folder path.')
    ap.add_argument('--save', type=str, default='./output/DeSite_BSU_result.json', help='Output path.')

    args = ap.parse_args()
    return args



def DeSite(map, threshold_region=0.8, threshold_binding=0.8, error=0.1):
    
    rows, cols = map.shape
    area = np.zeros_like(map, dtype=bool) # Define traversal area
    vvv = [] # Summit values
    ccc = [] # Coordinate positions
    region = []

    ## Remove smaller outliers (smaller outliers will cause the overall data distribution to be higher after normalization, which is not conducive to finding interaction sites)
    ## Example: 8GYM_L_C
    # Calculate mean, standard deviation, and set outlier range (k=2)
    mean = np.mean(map)
    std = np.std(map)
    k = 2  # Adjust k value to control outlier range
    lower_bound = mean - k * std
    upper_bound = mean + k * std

    # Set values below threshold to NaN (not participating in normalization)
    filtered_map = np.where(map < lower_bound, np.nan, map)


    # Normalization
    map_normalized = (filtered_map - np.nanmin(filtered_map)) / (np.nanmax(filtered_map) - np.nanmin(filtered_map))


    ## Find the first summit ##
    # Display coordinates and values
    coords = np.argwhere(~area)
    values = map_normalized[~area]

    # Maximum value and its coordinates
    binding_normalized_value = np.nanmax(values)
    binding_normalized_coord = tuple(coords[np.nanargmax(values)])


    ### When ASM maximum value does not exceed threshold_region, generate only one region around this maximum ###
    if map[binding_normalized_coord] < threshold_region:
        vv = [map[binding_normalized_coord]]
        cc = [binding_normalized_coord]
        rr = np.zeros_like(map, dtype=bool)

        ## Diffusion ##
        # Create queue
        queue = [binding_normalized_coord]

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            r, c = queue.pop(0)
            area[r, c] = True
            rr[r, c] = True

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                # Diffusion area should be within bounds and unexplored
                if 0 <= nr < rows and 0 <= nc < cols and not area[nr, nc]:

                    # When the value of the area to be diffused (nr,nc) is less than the previous point (r,c) + error, continue diffusion
                    if map_normalized[nr, nc] <= map_normalized[r, c] + error or np.isnan(map_normalized[nr, nc]):
                        queue.append((nr, nc))
                        area[nr, nc] = True
                        rr[nr, nc] = True

        rr_coords = np.argwhere(rr)
        rr_values = map[rr]
        rr_n_values = map_normalized[rr]
        # 1. Get sorting indices
        indices = np.argsort(rr_values)[::-1]
        # 2. Directly construct result list
        r_vv = []
        r_cc = []
        r_n_vv = []
        for i in indices:
            r_vv.append(float(rr_values[i]))  # Ensure it's Python float
            r_n_vv.append(float(rr_n_values[i]))
            r_cc.append((int(rr_coords[i][0]), int(rr_coords[i][1])))  # Ensure it's integer tuple

        llen = len([x for x in r_n_vv if x >= r_n_vv[0] * threshold_binding])
        vv = r_vv[:llen]
        cc = r_cc[:llen]

        if len(vv) != 0:
            vvv.append(vv)
            ccc.append(cc)

        region.append(rr)

    ### When ASM maximum value exceeds threshold_region, generate regions normally ###
    else:
        ### Find multiple summits, each summit diffuses in its neighboring area ###
        # while map[binding_normalized_coord] >= threshold_region:
        while map[binding_normalized_coord] >= threshold_region and map_normalized[binding_normalized_coord] >= 0.75:
            vv = [map[binding_normalized_coord]]
            cc = [binding_normalized_coord]
            rr = np.zeros_like(map, dtype=bool)

            ## Diffusion ##
            # Create queue
            queue = [binding_normalized_coord]

            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while queue:
                r, c = queue.pop(0)
                area[r, c] = True
                rr[r, c] = True

                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    # Diffusion area should be within bounds and unexplored
                    if 0 <= nr < rows and 0 <= nc < cols and not area[nr, nc]:

                        # When the value of the area to be diffused (nr,nc) is less than the previous point (r,c) + error, continue diffusion
                        if map_normalized[nr, nc] <= map_normalized[r, c] + error or np.isnan(map_normalized[nr, nc]):
                            queue.append((nr, nc))
                            area[nr, nc] = True
                            rr[nr, nc] = True


            rr_coords = np.argwhere(rr)
            rr_values = map[rr]
            rr_n_values = map_normalized[rr]
            # 1. Get sorting indices
            indices = np.argsort(rr_values)[::-1]
            # 2. Directly construct result list
            r_vv = []
            r_cc = []
            r_n_vv = []
            for i in indices:
                r_vv.append(float(rr_values[i]))  # Ensure it's Python float
                r_n_vv.append(float(rr_n_values[i]))
                r_cc.append((int(rr_coords[i][0]), int(rr_coords[i][1])))  # Ensure it's integer tuple

            llen = len([x for x in r_n_vv if x >= r_n_vv[0] * threshold_binding])
            vv = r_vv[:llen]
            cc = r_cc[:llen]

            if len(vv) != 0:
                vvv.append(vv)
                ccc.append(cc)

            region.append(rr)


            ### Reset
            if np.any(~area):
                ## Reset summit ##
                # Reset display coordinates and values
                coords = np.argwhere(~area)
                values = map_normalized[~area]

                # Subsequent maximum value and its coordinates
                binding_normalized_value = np.nanmax(values)
                binding_normalized_coord = tuple(coords[np.nanargmax(values)])
            else:
                break
        
    return vvv, ccc, region



def BSU(vvv):
    if len(vvv) == 0:
        return 0., []

    else:
        regions = []
        for values in vvv:
            n = len(values)
            vv = np.array(values)
            v = np.sort(vv[vv>=0.1]*10)[::-1] 
            bs = np.log10(sum((v**12 - v**6))*1) + np.log10(sum((v**2))*100)

            regions.append(bs)
        
        return max(regions), regions



def main():
    args = parser()

    results = {}
    ASMlist = os.listdir(args.ASMfolder)

    for id in ASMlist:
        asm = np.load(f'{args.ASMfolder}{id}')[0]
        vvv, ccc, region = DeSite(asm, threshold_region=0.8, threshold_binding=0.8, error=0.1)

        bsu, regions = BSU(vvv)

        results[id] = {'site values': vvv, 'site coordinates': ccc, 'BSU': bsu, 'BSU rank': regions}

    with open(args.save, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()