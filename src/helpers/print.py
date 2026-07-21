def print_in_box(content):
    title_text = content['title']
    print(f"+{title_text.center(98, '-')}+", end='\n')
    
    for line in content['lines']:
        print(f"|{line.center(98)}|", end='\n')
        
    end_text = '-'*98
    print(f"+{end_text.center(98)}+", end='\n')

def finish_simulation_log(file_name, time):
    print(f"\n ✅✅✅  Successfully Complete Simulation in {time:.2f} seconds ✅✅✅  \n ".center(100))
    final_msg = " \n Saved simulation data: \n \
            { \n \
                \"metadata\" : dict[dt, nlat, nlon, lmax, mmax, grid, step_per_save], \n \
                \"trajectory\" : tensor(N, 3, lmax, mmax)= [spectral(geopotential, vorticity, divergence)]_N  \n \
            } \n \
            " + f"\n ---> {file_name}"
    print(final_msg)