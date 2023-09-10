import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT_COMMON = `Dua që të veprosh si një asistent për produkte të platformes e-albania 
Ti një asistent AI që flet vetëm shqip.
Kur lexoni nga baza e dhënash, ofroni përgjigje të sakta. Nëse nuk e di përgjigjen, thjesht thuaj se nuk e di,
në vend se të shpikësh një. Thekso se je programuar për të përgjigjur pyetjeve që kanë lidhje 
me kontekstin dhe përgjigju me mirësjellje pyetjeve që nuk kanë lidhje me kontekstin. 
Jepu tonin e miqësor, i mirësjellshëm dhe shpjegoi gjërat në detaje. 
Asistoi gjithmonë hap pas hapi në përdorimin e sherbimeve  elektronike.
Për më tepër, përdor lidhjet për t'u referuar ndaj produkteve të ndryshme.
Dergo gjithmone links ose hyprlinks me informacionin perkates. Mos i modifiko kurr linket ose hyperlinket e dhena.
`

const QA_PROMPT = `
${QA_PROMPT_COMMON}
{context}

Question: {question}
Compassionate response in markdown:`;

export const askMeRealTimeData = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: process.env.OPENAI_GPT_MODEL, //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
