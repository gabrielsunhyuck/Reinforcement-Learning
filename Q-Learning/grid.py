class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, a):
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_down()

        reward  = -1
        done    = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        ''' 장애물이나 벽이 있는 경우 모두 pass '''
        if self.y == 1 and self.x in [0, 1, 2]:
            pass
        elif self.y == 3 and self.x in [2, 3, 4]:
            pass
        elif self.y == 6:
            pass
        else:
            self.y += 1

    def move_left(self):
        ''' 장애물이나 벽이 있는 경우 모두 pass '''
        if self.y == 0:
            pass
        elif self.y == 3 and self.x in [0, 1, 2]:
            pass
        elif self.y == 5 and self.x in [2, 3, 4]:
            pass
        else:
            self.y -= 1
        
    def move_up(self):
        ''' 장애물이나 벽이 있는 경우 모두 pass '''
        if self.x == 0:
            pass
        elif self.x == 3 and self.y == 2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        ''' 장애물이나 벽이 있는 경우 모두 pass '''
        if self.x == 4:
            pass
        elif self.x == 1 and self.y == 4:
            pass
        else:
            self.x += 1

    def is_done(self):
        if self.x == 4 and self.y == 6:
            return True
        else: return False

    def get_state(self):
        return (self.x, self.y)
    
    def reset(self):
        self.x = 0
        self.y = 0
        return self.x, self.y