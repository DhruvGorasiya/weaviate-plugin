from typing import Any
import weaviate
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
try:
    from ..utils.validators import validate_weaviate_url, validate_api_key
except ImportError:
    from utils.validators import validate_weaviate_url, validate_api_key

class WeaviatePluginProvider(ToolProvider):
    
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            url = credentials.get('url', '')
            api_key = credentials.get('api_key', '')
            
            if not validate_weaviate_url(url):
                raise ValueError("Invalid Weaviate URL format")
            
            if api_key and not validate_api_key(api_key):
                raise ValueError("Invalid API key format")
            
            auth_config = weaviate.AuthApiKey(api_key=api_key) if api_key else None
            
            client = weaviate.connect_to_local(
                url=url,
                auth_credentials=auth_config,
                timeout_config=(5, 10)
            )
            
            client.close()
            
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))

    #########################################################################################
    # If OAuth is supported, uncomment the following functions.
    # Warning: please make sure that the sdk version is 0.4.2 or higher.
    #########################################################################################
    # def _oauth_get_authorization_url(self, redirect_uri: str, system_credentials: Mapping[str, Any]) -> str:
    #     """
    #     Generate the authorization URL for weaviate_plugin OAuth.
    #     """
    #     try:
    #         """
    #         IMPLEMENT YOUR AUTHORIZATION URL GENERATION HERE
    #         """
    #     except Exception as e:
    #         raise ToolProviderOAuthError(str(e))
    #     return ""
        
    # def _oauth_get_credentials(
    #     self, redirect_uri: str, system_credentials: Mapping[str, Any], request: Request
    # ) -> Mapping[str, Any]:
    #     """
    #     Exchange code for access_token.
    #     """
    #     try:
    #         """
    #         IMPLEMENT YOUR CREDENTIALS EXCHANGE HERE
    #         """
    #     except Exception as e:
    #         raise ToolProviderOAuthError(str(e))
    #     return dict()

    # def _oauth_refresh_credentials(
    #     self, redirect_uri: str, system_credentials: Mapping[str, Any], credentials: Mapping[str, Any]
    # ) -> OAuthCredentials:
    #     """
    #     Refresh the credentials
    #     """
    #     return OAuthCredentials(credentials=credentials, expires_at=-1)
