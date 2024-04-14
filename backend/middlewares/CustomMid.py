from django.utils.deprecation import MiddlewareMixin

class AddCustomHeadersMiddleware(MiddlewareMixin):
    def process_response(self, _, response):
        response['Access-Control-Expose-Headers'] = 'X-Error-Message'
        return response