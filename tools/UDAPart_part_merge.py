 # for bus
part_idxs = [[0], [1], [2], [3], # wheel_front_left, wheel_front_right, wheel_back_left, wheel_back_right
                    [4], [5], [6], [7], # door_front_left, door_front_right, door_mid_left, door_mid_right
                    [8], [9], [10,11], [12,13], # door_back_left, door_back_right, window_front_left, window_front_right 
                    [14,15,16,17], [18,19,20,21], [22], [23], # window_back_left,window_back_right,licplate_front,licplate_back
                    [24], [25], [26], [27], # windshield_front, windshield_back, head_light_left, head_light_right
                    [28], [29], [30], [31], # tail_light_left, tail_light_right, mirror_left, mirror_right
                    [32], [33], [34], [35], # bumper_front, bumper_back, trunk, roof
                    [36], [37], [38], [39]] # frame_front, frame_back, frame_left, frame_right

 # for bicycle
part_idxs = [[0], [1], [2], [3], # wheel_front, wheel_back, fender_front, fender_back
                    [4], [5,13], [6,14], [7,15], # fork, handle_left, handle_right, saddle
                    [8,16], [9,17], [10,18], [11,19], # drive_chain, pedal_left, pedal_right, crank_arm_left
                    [12,20], [21], [22], [23], # crank_arm_right, carrier, rearlight, side_stand
                    [24,25,26]] # frame

# for motorbike
part_idxs = [[0], [1], [2], [3], [4,5,6,14,27,28,29,30], # wheel_front, wheel_back, fender_front, fender_back, frame
            [7], [8], [9], # mirror_left, mirror_right, windscreen
            [10], [11], [12], [13], # license_plate, seat, seat_back, gas_tank
            [15], [16], [17,18,19], # handle_left, handle_right, headlight
            [20,21,22], [23,24], # taillight, exhaust_left
            [25,26], # exhaust_right
            [31], [32], [33,34]] # engine, cover_front, cover_body

# for aeroplane
part_idxs = [[0], [1,2,3,4,5,6,7,8,9], # propeller, cockpit
            [10], [11], [12], # wing_left, wing_right, fin
            [13], [14], [15], [16], # tailplane_left, tailplane_right, wheel_front, landing_gear_front
            [17], [18], [19], [20], # wheel_back_left, gear_back_left, wheel_back_right, gear_back_right
            [21], [22], [23,24,25,26], [27,28,29,30], # engine_left, engine_right, door_left, door_right
            [31], [32], # bomb_left, bomb_right
            [33,34], [35,36], [37,38]] # window_left, window_right, body




# for car   label 0 - 30
# back_bumper
# back_left_door
# back_left_wheel
# back_left_window
# back_license_plate
# back_right_door
# back_right_wheel
# back_right_window
# back_windshield
# front_bumper
# front_left_door
# front_left_wheel
# front_left_window
# front_license_plate
# front_right_door
# front_right_wheel
# front_right_window
# front_windshield
# hood
# left_frame
# left_head_light
# left_mirror
# left_quarter_window
# left_tail_light
# right_frame
# right_head_light
# right_mirror
# right_quarter_window
# right_tail_light
# roof
# trunk


# The above are merging UDAPart's annotated 3D parts to correspond its real images annotations


# for bus -- 3 parts
part_idxs = [[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33,34,35,36,37,38,39],
            [0,1,2,3],
            [30, 31]]
# for car -- 3 parts
part_idxs = [[0, 1, 3, 4, 5,7,8,9,10,12,13,14,16,17,18,19,20,22,23,24,25, 27,28,29,30],
            [2, 6, 11, 15],
            [21, 26]]

# # for bike -- 4 parts: body, head (handler), seat, tire
part_idxs = [[2,3,4,8,9,10,11,12,16,17,18,19,20,21,22,23,24,25,26],
            [5,6,13,14],
            [7,15],
            [0,1]]