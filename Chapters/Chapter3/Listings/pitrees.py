for room in black_board.task_list.keys():
    # Convert the room name to upper case for consistency
    ROOM = room.upper()

    # Initialize the CLEANING_ROUTINE selector for this room
    CLEANING_ROUTINE[room] = Selector("CLEANING_ROUTINE_" + ROOM)

    # Initialize the CHECK_ROOM_CLEAN condition
    CHECK_ROOM_CLEAN[room] = CheckRoomCleaned(room)

    # Add the CHECK_ROOM_CLEAN condition to the CLEANING_ROUTINE selector
    CLEANING_ROUTINE[room].add_child(CHECK_ROOM_CLEAN[room])

    # Initialize the CLEAN_ROOM sequence for this room
    CLEAN_ROOM[room] = Sequence("CLEAN_" + ROOM)

    # Initialize the NAV_ROOM selector for this room
    NAV_ROOM[room] = Selector("NAV_ROOM_" + ROOM)

    # Initialize the CHECK_LOCATION condition for this room
    CHECK_LOCATION[room] = CheckLocation(room, self.room_locations)

    # Add the CHECK_LOCATION condition to the NAV_ROOM selector
    NAV_ROOM[room].add_child(CHECK_LOCATION[room])

    # Add the MOVE_BASE task for this room to the NAV_ROOM selector
    NAV_ROOM[room].add_child(MOVE_BASE[room])

    # Add the NAV_ROOM selector to the CLEAN_ROOM sequence
    CLEAN_ROOM[room].add_child(NAV_ROOM[room])

    # Initialize the TASK_LIST iterator for this room
    TASK_LIST[room] = Iterator("TASK_LIST_" + ROOM)

    # Add the tasks assigned to this room
    for task in black_board.task_list[room]:
        # Initialize the DO_TASK sequence for this room and task
        DO_TASK = Sequence("DO_TASK_" + ROOM + "_" + task.name)

        # Add a CHECK_LOCATION condition to the DO_TASK sequence
        DO_TASK.add_child(CHECK_LOCATION[room])

        # Add the task itself to the DO_TASK sequence
        DO_TASK.add_child(task)

        # Create an UPDATE_TASK_LIST task for this room and task
        UPDATE_TASK_LIST[room + "_" + task.name] = UpdateTaskList(room, task)

        # Add the UPDATE_TASK_LIST task to the DO_TASK sequence
        DO_TASK.add_child(UPDATE_TASK_LIST[room + "_" + task.name])

        # Add the DO_TASK sequence to the TASK_LIST iterator
        TASK_LIST[room].add_child(DO_TASK)

    # Add the room TASK_LIST iterator to the CLEAN_ROOM sequence
    CLEAN_ROOM[room].add_child(TASK_LIST[room])

    # Add the CLEAN_ROOM sequence to the CLEANING_ROUTINE selector
    CLEANING_ROUTINE[room].add_child(CLEAN_ROOM[room])

    # Add the CLEANING_ROUTINE for this room to the CLEAN_HOUSE sequence
    CLEAN_HOUSE.add_child(CLEANING_ROUTINE[room])
