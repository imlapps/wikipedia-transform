from dagster import ConfigurableResource
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI

from etl.models import OpenAiResourceParams, Record, wikipedia
from etl.models.types import (
    EnrichmentType,
    ModelQuestion,
    ModelResponse,
    RecordKey,
    RecordType,
)


class OpenAiGenerativeModelResource(ConfigurableResource):  # type: ignore
    """A ConfigurableResource to enrich Records using OpenAI's generative AI models."""

    openai_resource_params: OpenAiResourceParams
    __template = """
                Keep the answer as concise as possible.
                Question: {question}
                """

    def __create_question(self, record_key: RecordKey) -> ModelQuestion:
        """Return a question for an OpenAI model."""

        match self.openai_resource_params.enrichment_type:
            case EnrichmentType.SUMMARY:
                return f"In 5 sentences, give a summary of {record_key} based on {record_key}'s Wikipedia entry."

    def __create_chat_model(self) -> ChatOpenAI:
        """Return an OpenAI chat model."""

        return ChatOpenAI(
            name=str(self.openai_resource_params.openai_settings.generative_model_name),
            temperature=self.openai_resource_params.openai_settings.temperature,
        )

    def __build_chain(self, model: ChatOpenAI) -> RunnableSerializable:
        """Build a chain that consists of an OpenAI prompt, large language model and an output parser."""

        prompt = PromptTemplate.from_template(self.__template)

        return {"question": RunnablePassthrough()} | prompt | model | StrOutputParser()

    def __generate_response(
        self, *, question: ModelQuestion, chain: RunnableSerializable
    ) -> ModelResponse:
        """Invoke the OpenAI large language model and generate a response."""

        return chain.invoke(question)

    def enrich_record(self, record: Record) -> Record:
        """
        Return a Record that has been enriched using OpenAI models.

        Return the original Record if OpenAiResourceParams.enrichment_type is not a field of Record.
        """

        if self.openai_resource_params.enrichment_type in record.model_fields.keys():
            match self.openai_resource_params.record_type:
                case RecordType.WIKIPEDIA:
                    return wikipedia.Article(
                        **(
                            record.model_dump(by_alias=True)
                            | {
                                self.openai_resource_params.enrichment_type: self.__generate_response(
                                    question=self.__create_question(
                                        record_key=record.key
                                    ),
                                    chain=self.__build_chain(
                                        self.__create_chat_model()
                                    ),
                                )
                            }
                        )
                    )
        return record
