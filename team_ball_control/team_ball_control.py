class TeamBallControl:
    def __init__(self):
        self.sequence = []
        self.last_team = None

    def update(self, assigned_player, players_frame):
        if assigned_player != -1:
            player_data = players_frame.get(assigned_player, {})
            team = player_data.get('team')  # ← None kalau tidak ada key 'team'
            
            if team is not None:
                self.last_team = team
                self.sequence.append(team)
            else:
                # Player ditemukan tapi belum ada team assignment
                self.sequence.append(self.last_team)
        else:
            self.sequence.append(self.last_team)

    def get_sequence(self):
        return self.sequence