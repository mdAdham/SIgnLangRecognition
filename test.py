from requests import get
from requests import post

# url = "http://192.168.178.118:8123/api/services/media_player/play_media"
# headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI3NzUyMDQzZmNmYTk0YmJkODRiZTk4Zjg5MDFlYzcxMSIsImlhdCI6MTY4ODM3MzM2OSwiZXhwIjoyMDAzNzMzMzY5fQ.UKI8FsPAtFJBL9lZzcQ3LhJbRkaLyzZ2gU7l_sYFZq0"}
# data = {"entity_id": "media_player.imrans_echo_dot","media_content_id": "Start", "media_content_type": "custom"}
# # data = {"entity_id": "select.siemens_ti9555x1de_68a40e325683_bsh_common_setting_powerstate"}


# response = post(url, headers=headers, json=data)
# print(response.text)

url = "http://192.168.178.118:8123/api/services/media_player/play_media"
headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI3NzUyMDQzZmNmYTk0YmJkODRiZTk4Zjg5MDFlYzcxMSIsImlhdCI6MTY4ODM3MzM2OSwiZXhwIjoyMDAzNzMzMzY5fQ.UKI8FsPAtFJBL9lZzcQ3LhJbRkaLyzZ2gU7l_sYFZq0"}
data = {"entity_id": "media_player.imrans_echo_dot", "media_content_id": "Stop", "media_content_type": "custom"}
response = post(url, headers=headers, json=data)
print(response.text)

