from requests import get
from requests import post

url = "http://192.168.178.118:8123/api/services/homeassistant/turn_on"
headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZGU1YWQ5NjI1MDQ0NWE3YTI1NGRjYjM0NDAzYWU4MyIsImlhdCI6MTY4ODAzNjM0OSwiZXhwIjoyMDAzMzk2MzQ5fQ.s-GU1OqsLPW7HYo2ZkSGg6twERApfqIl7W0gjJraq20"}
data = {"entity_id": "light.haso_bett_led", "brightness": 254, "rgb_color": [255, 0, 0], "transition":"5"}
# data = {"entity_id": "select.siemens_ti9555x1de_68a40e325683_bsh_common_setting_powerstate"}


response = post(url, headers=headers, json=data)
print(response.text)