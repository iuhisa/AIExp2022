import pexelsPy
import requests

PEXELS_API = '563492ad6f91700001000001c85472c49d3a4c189a1ed7baa64e4ae5'
api = pexelsPy.API(PEXELS_API)

pageNumbers = 1 
resultsPage = 30

api.search_videos('nature', page=pageNumbers, results_per_page=resultsPage)
videos = api.get_videos()


for data in videos:
    url_video = 'https://www.pexels.com/video/' + str(data.id) + '/download'
    r = requests.get(url_video)
    with open(data.url.split('/')[-2]+'.mp4', 'wb') as outfile:
        outfile.write(r.content)
