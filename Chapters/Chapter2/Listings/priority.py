for node in children:
    status = node.tick()

    if status != False:
        return status

return False
